"""Behaviour cloning policy (PyTorch).

    features (+ optional camera / screen images) -> MLP -> [waypoint offset (cm), finger]
    -> clipped step toward the waypoint (same rule as the expert)

Image inputs go through a small CNN encoder each (the "vision encoder" of the
architecture); by default the policy uses only the low-dimensional features, which train in
seconds on a CPU. Enable images with `--images` for both collection and training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from learning.dataset import FEATURE_DIM, Demonstrations, decode_target, low_dim_features


def pick_device(device: str = "auto") -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


class VisionEncoder(nn.Module):
    """uint8 (B, H, W, 3) image -> (B, out_dim) embedding."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(64 * 16, out_dim), nn.ReLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image.permute(0, 3, 1, 2).float() / 255.0
        return self.net(x)


class BCNetwork(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM, image_keys: tuple[str, ...] = (), hidden: int = 256,
                 embed_dim: int = 64):
        super().__init__()
        self.image_keys = tuple(image_keys)
        self.encoders = nn.ModuleDict({k: VisionEncoder(embed_dim) for k in self.image_keys})
        in_dim = feature_dim + embed_dim * len(self.image_keys)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, features: torch.Tensor, images: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        parts = [features] + [self.encoders[k](images[k]) for k in self.image_keys]
        out = self.mlp(torch.cat(parts, dim=1))
        return torch.cat([out[:, :3], torch.sigmoid(out[:, 3:])], dim=1)


def train_bc(
    demos: Demonstrations,
    out_path: str | Path,
    epochs: int = 40,
    batch_size: int = 512,
    lr: float = 1e-3,
    use_images: bool = False,
    device: str = "auto",
    val_fraction: float = 0.1,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Fit a BCNetwork to demonstrations; returns the training history."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = pick_device(device)
    image_keys = tuple(sorted(demos.images)) if use_images else ()
    if use_images and not image_keys:
        raise ValueError("demonstrations contain no images; collect with --images")

    # Validation split by episode so validation states are not neighbours of training states.
    episodes = np.unique(demos.episode)
    n_val = max(1, int(len(episodes) * val_fraction)) if len(episodes) > 1 else 0
    val_eps = rng.choice(episodes, n_val, replace=False) if n_val else np.array([], int)
    is_val = np.isin(demos.episode, val_eps)
    train_idx, val_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)

    feats = torch.from_numpy(demos.features).to(dev)
    targets = torch.from_numpy(demos.targets).to(dev)
    imgs = {k: torch.from_numpy(demos.images[k]).to(dev) for k in image_keys}

    net = BCNetwork(demos.features.shape[1], image_keys).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    # Smooth L1 keeps millimetre precision near the goal; the finger command (range 0-0.75)
    # is weighted up so it matters as much as the motion.
    weights = torch.tensor([1.0, 1.0, 1.0, 4.0], device=dev)

    def loss_on(idx: np.ndarray) -> torch.Tensor:
        i = torch.from_numpy(idx).to(dev)
        pred = net(feats[i], {k: v[i] for k, v in imgs.items()})
        err = nn.functional.smooth_l1_loss(pred, targets[i], reduction="none", beta=0.05)
        return (err * weights).mean()

    history = {"train": [], "val": []}
    for epoch in range(epochs):
        net.train()
        perm = rng.permutation(train_idx)
        total = 0.0
        for start in range(0, len(perm), batch_size):
            batch = perm[start : start + batch_size]
            loss = loss_on(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        sched.step()
        history["train"].append(total / max(len(perm), 1))
        net.eval()
        with torch.no_grad():
            val = loss_on(val_idx).item() if len(val_idx) else float("nan")
        history["val"].append(val)
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  train {history['train'][-1]:.5f}  val {val:.5f}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "feature_dim": demos.features.shape[1],
            "image_keys": image_keys,
            "image_shapes": {k: tuple(demos.images[k].shape[1:]) for k in image_keys},
            "history": history,
            "num_samples": len(demos),
            "task_specs": demos.task_specs,
        },
        out_path,
    )
    if verbose:
        print(f"saved {out_path} ({len(train_idx)} train / {len(val_idx)} val samples)")
    return history


class BCPolicy:
    """Runs a trained BCNetwork in the environment."""

    name = "bc"

    def __init__(self, checkpoint: str | Path, max_step: float, device: str = "auto"):
        self.device = pick_device(device)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.image_keys = tuple(ckpt["image_keys"])
        self.image_shapes = ckpt.get("image_shapes", {})
        self.net = BCNetwork(ckpt["feature_dim"], self.image_keys).to(self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.max_step = max_step
        self.skill = "learned"

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def act(self, obs: dict) -> np.ndarray:
        feats = torch.from_numpy(low_dim_features(obs)).unsqueeze(0).to(self.device)
        images = {}
        for k in self.image_keys:
            if k not in obs:
                raise KeyError(f"policy needs observation {k!r}; enable render_obs / obs_cameras")
            if self.image_shapes and tuple(obs[k].shape) != tuple(self.image_shapes[k]):
                raise ValueError(f"{k}: policy trained on {self.image_shapes[k]}, got {obs[k].shape}")
            images[k] = torch.from_numpy(np.ascontiguousarray(obs[k])).unsqueeze(0).to(self.device)
        target = self.net(feats, images)[0].cpu().numpy()
        return decode_target(target, self.max_step)
