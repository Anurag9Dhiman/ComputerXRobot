"""Behaviour cloning on the vision track (benchmark/tracks.py).

Inputs, all available on the vision track:

    robot_cam, keyboard_cam,     box-downsampled from the track's resolution (5x, 4x, 5x, 5x):
    mouse_cam, screen_cam        48x64, 64x128, 64x64, 80x128
    proprioception               hand target and tracking error, finger, joint positions and
                                 velocities, palm touch
    instruction                  the task instruction, character by character

The diagnostic variant (`subgoal=True`, evaluated on the `vision_subgoal` track) also gets the
task's next sub-goal, as the state track does: the character to type, or that a click is due,
with which button and where the target is on screen. It removes planning and reading, and keeps
perception for control (finding keys, the mouse, the cursor), so the two variants separate the
two sources of difficulty.

Network: a CNN per camera ending in a spatial soft-argmax (keypoint coordinates keep the
position precision a key press or a click needs) plus average pooling; a bidirectional GRU over
the instruction's characters; an MLP head. Labels and decoding are those of the state-track BC
policy (learning/dataset.py): the expert's waypoint offset (cm) and finger command. Training
augments every camera with random shifts (up to 4 px) and brightness / contrast jitter.

    python -m benchmark collect --split train --per-task 40 --vision --out data/vision_train.npz
    python -m learning.vision train --data data/vision_train.npz --seed 0 --out checkpoints/vbc_s0.pt
    python -m benchmark run --agent vbc:checkpoints/vbc_s0.pt --all-test --workers 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from environment.browser import SCREEN_H, SCREEN_W
from environment.keyboard_layout import KEYS
from learning.dataset import (CHAR_INDEX, WORKSPACE_CENTER, WORKSPACE_SCALE, Demonstrations, decode_target)

CAMERAS = {"robot_cam": 5, "keyboard_cam": 4, "mouse_cam": 5, "screen_cam": 5}  # camera -> downsampling
PROPRIO_DIM = 3 + 3 + 1 + 1 + 6 + 6
SUBGOAL_DIM = len(KEYS) + 4
MAX_INSTRUCTION = 96  # characters
VOCAB = 97  # 0 = padding, 1-95 = printable ASCII, 96 = anything else
SHIFT = 4  # px, augmentation
RECORD_EVERY = 2  # the collector records every second step (neighbouring frames are near-duplicates)


# ------------------------------------------------------------------ inputs
def downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """Box filter (mean over factor x factor blocks, rounded), uint8 in and out."""
    h, w = image.shape[0] // factor, image.shape[1] // factor
    blocks = image[: h * factor, : w * factor].reshape(h, factor, w, factor, 3).astype(np.uint16).sum((1, 3))
    return ((blocks + factor * factor // 2) // (factor * factor)).astype(np.uint8)


def proprio(obs: dict) -> np.ndarray:
    return np.concatenate([
        (obs["tool_target"] - WORKSPACE_CENTER) / WORKSPACE_SCALE,
        (obs["tool_pos"] - obs["tool_target"]) * 100.0,
        [obs["finger"], float(obs["palm_on_mouse"])],
        obs["qpos"],
        np.asarray(obs["qvel"]) * 0.1,
    ]).astype(np.float32)


def subgoal(obs: dict) -> np.ndarray:
    f = np.zeros(SUBGOAL_DIM, np.float32)
    if obs.get("expected_char") is not None:
        f[CHAR_INDEX[obs["expected_char"]]] = 1.0
    target = obs.get("click_target_px")
    if target is not None:
        f[len(KEYS)] = 1.0
        f[len(KEYS) + 1] = float(obs.get("click_button") == "right")
        f[len(KEYS) + 2:] = np.asarray(target, float) / [SCREEN_W / 2, SCREEN_H / 2] - 1.0
    return f


def encode_instruction(text: str) -> np.ndarray:
    ids = [ord(c) - 31 if 32 <= ord(c) <= 126 else VOCAB - 1 for c in text[:MAX_INSTRUCTION]]
    out = np.zeros(MAX_INSTRUCTION, np.int16)
    out[: len(ids)] = ids
    return out


def vision_inputs(obs: dict) -> dict[str, np.ndarray]:
    """Everything the policy uses from one vision-track observation (plus the sub-goal, which
    only the diagnostic variant reads). Recorded per step by the collector."""
    out = {cam: downsample(obs[cam], f) for cam, f in CAMERAS.items()}
    out["proprio"] = proprio(obs)
    out["instruction"] = encode_instruction(obs["instruction"])
    out["subgoal"] = subgoal(obs)
    return out


# ------------------------------------------------------------------ network
class SpatialSoftArgmax(nn.Module):
    """(B, C, H, W) -> (B, 2C): expected (x, y) in [-1, 1] of each channel's softmax."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = torch.softmax(x.flatten(2), dim=-1).view(b, c, h, w)
        xs = torch.linspace(-1, 1, w, device=x.device)
        ys = torch.linspace(-1, 1, h, device=x.device)
        return torch.cat([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], dim=1)


class CameraEncoder(nn.Module):
    def __init__(self, keypoints: int = 32, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
        )
        self.keypoints = nn.Conv2d(64, keypoints, 1)
        self.argmax = SpatialSoftArgmax()
        self.out = nn.Sequential(nn.Linear(2 * keypoints + 64, out_dim), nn.ReLU())

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # float (B, 3, H, W) in [0, 1]
        h = self.conv(image)
        return self.out(torch.cat([self.argmax(self.keypoints(h)), h.mean((2, 3))], dim=1))


class InstructionEncoder(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, 32, padding_idx=0)
        self.gru = nn.GRU(32, out_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        lengths = (ids > 0).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.embed(ids), lengths, batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)  # (2, B, out/2)
        return torch.cat([h[0], h[1]], dim=1)


class VisionBCNetwork(nn.Module):
    def __init__(self, use_subgoal: bool = False, hidden: int = 512):
        super().__init__()
        self.use_subgoal = use_subgoal
        self.cameras = nn.ModuleDict({cam: CameraEncoder() for cam in CAMERAS})
        self.instruction = InstructionEncoder()
        in_dim = 128 * len(CAMERAS) + 128 + PROPRIO_DIM + (SUBGOAL_DIM if use_subgoal else 0)
        self.head = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 4))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """`instruction` holds the batch's distinct instructions and `instruction_index` which one
        each sample has (see `unique_instructions`): a batch has only a few distinct ones."""
        parts = [self.cameras[cam](batch[cam]) for cam in CAMERAS]
        parts += [self.instruction(batch["instruction"])[batch["instruction_index"]], batch["proprio"]]
        if self.use_subgoal:
            parts.append(batch["subgoal"])
        out = self.head(torch.cat(parts, dim=1))
        return torch.cat([out[:, :3], torch.sigmoid(out[:, 3:])], dim=1)


def unique_instructions(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(distinct instruction rows, index of each sample's row), computed on the CPU."""
    uniq, index = torch.unique(ids.cpu(), dim=0, return_inverse=True)
    return uniq.long(), index


def _to_float(images: torch.Tensor) -> torch.Tensor:
    return images.permute(0, 3, 1, 2).float() / 255.0


def augment(images: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random shift (replicate-padded crop) and brightness / contrast jitter, per sample."""
    b, _, h, w = images.shape
    dev = images.device
    padded = nn.functional.pad(images, (SHIFT, SHIFT, SHIFT, SHIFT), mode="replicate")
    dx = torch.randint(0, 2 * SHIFT + 1, (b,), generator=gen).to(dev)
    dy = torch.randint(0, 2 * SHIFT + 1, (b,), generator=gen).to(dev)
    rows = (dy[:, None] + torch.arange(h, device=dev)[None])[:, :, None].expand(b, h, w)
    cols = (dx[:, None] + torch.arange(w, device=dev)[None])[:, None, :].expand(b, h, w)
    idx = torch.arange(b, device=dev)[:, None, None].expand(b, h, w)
    shifted = padded.permute(0, 2, 3, 1)[idx, rows, cols].permute(0, 3, 1, 2)
    brightness = (torch.rand(b, 1, 1, 1, generator=gen) * 0.4 - 0.2).to(dev)
    contrast = (torch.rand(b, 1, 1, 1, generator=gen) * 0.4 + 0.8).to(dev)
    mean = shifted.mean((1, 2, 3), keepdim=True)
    return ((shifted - mean) * contrast + mean + brightness).clamp(0, 1)


# ------------------------------------------------------------------ training
def train_vision_bc(demos: Demonstrations, out_path: str | Path, use_subgoal: bool = False, epochs: int = 40,
                    batch_size: int = 128, lr: float = 3e-4, seed: int = 0, device: str = "auto",
                    val_fraction: float = 0.05, verbose: bool = True) -> dict:
    missing = [k for k in (*CAMERAS, "proprio", "instruction", "subgoal") if k not in demos.images]
    if missing:
        raise ValueError(f"demonstrations lack {missing}; collect them with --vision")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    episodes = np.unique(demos.episode)
    val_eps = rng.choice(episodes, max(1, int(len(episodes) * val_fraction)), replace=False)
    is_val = np.isin(demos.episode, val_eps)
    train_idx, val_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    data = {k: torch.from_numpy(np.ascontiguousarray(demos.images[k])) for k in
            (*CAMERAS, "proprio", "instruction", "subgoal")}  # stays on the CPU; batches move
    targets = torch.from_numpy(demos.targets)

    net = VisionBCNetwork(use_subgoal).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(train_idx) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    weights = torch.tensor([1.0, 1.0, 1.0, 4.0], device=dev)

    def loss_on(idx: np.ndarray, train: bool) -> torch.Tensor:
        i = torch.from_numpy(idx)
        batch = {k: data[k][i].to(dev) for k in data if k != "instruction"}
        uniq, index = unique_instructions(data["instruction"][i])
        batch["instruction"], batch["instruction_index"] = uniq.to(dev), index.to(dev)
        for cam in CAMERAS:
            batch[cam] = _to_float(batch[cam])
            if train:
                batch[cam] = augment(batch[cam], gen)
        err = nn.functional.smooth_l1_loss(net(batch), targets[i].to(dev), reduction="none", beta=0.05)
        return (err * weights).mean()

    history = {"train": [], "val": []}
    t0 = time.time()
    for epoch in range(epochs):
        net.train()
        perm, total = rng.permutation(train_idx), 0.0
        for start in range(0, len(perm), batch_size):
            batch = perm[start: start + batch_size]
            loss = loss_on(batch, True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(batch)
        history["train"].append(total / len(perm))
        net.eval()
        with torch.no_grad():
            vals = [loss_on(val_idx[s: s + 512], False).item() * len(val_idx[s: s + 512])
                    for s in range(0, len(val_idx), 512)]
        history["val"].append(sum(vals) / max(len(val_idx), 1))
        if verbose:
            print(f"  epoch {epoch:3d}  train {history['train'][-1]:.4f}  val {history['val'][-1]:.4f}  "
                  f"{time.time() - t0:6.0f}s", flush=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "vision_bc", "state_dict": net.state_dict(), "use_subgoal": use_subgoal,
                "cameras": CAMERAS, "history": history, "num_samples": len(demos), "seed": seed,
                "epochs": epochs, "task_specs": demos.task_specs}, out_path)
    if verbose:
        print(f"saved {out_path} ({len(train_idx)} train / {len(val_idx)} val samples)")
    return history


# ------------------------------------------------------------------ policy
class VisionBCPolicy:
    def __init__(self, checkpoint: str | Path, max_step: float, device: str = "cpu"):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        if ckpt.get("kind") != "vision_bc":
            raise ValueError(f"{checkpoint} is not a vision-BC checkpoint")
        if ckpt["cameras"] != CAMERAS:
            raise ValueError(f"checkpoint cameras {ckpt['cameras']} differ from {CAMERAS}")
        self.device = torch.device(device)
        self.use_subgoal = bool(ckpt["use_subgoal"])
        self.net = VisionBCNetwork(self.use_subgoal).to(self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.max_step = max_step

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def act(self, obs: dict) -> np.ndarray:
        if self.use_subgoal and "expected_char" not in obs:
            raise KeyError("this policy needs the next sub-goal: evaluate it on the vision_subgoal track")
        x = vision_inputs(obs)
        batch = {cam: _to_float(torch.from_numpy(x[cam])[None]).to(self.device) for cam in CAMERAS}
        batch["proprio"] = torch.from_numpy(x["proprio"])[None].to(self.device)
        batch["instruction"] = torch.from_numpy(x["instruction"].astype(np.int64))[None].to(self.device)
        batch["instruction_index"] = torch.zeros(1, dtype=torch.long, device=self.device)
        batch["subgoal"] = torch.from_numpy(x["subgoal"])[None].to(self.device)
        return decode_target(self.net(batch)[0].cpu().numpy(), self.max_step)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train a vision-BC policy")
    t.add_argument("--data", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--subgoal", action="store_true", help="diagnostic variant: also condition on the next sub-goal")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    demos = Demonstrations.concatenate([Demonstrations.load(d) for d in args.data])
    print(f"{len(demos)} samples from {len(demos.task_specs)} episodes")
    train_vision_bc(demos, args.out, args.subgoal, args.epochs, args.batch_size, args.lr, args.seed, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
