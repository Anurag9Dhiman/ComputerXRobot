"""Recurrent behaviour cloning: a GRU over the whole episode (`bc-gru`).

The hard tier hides the dynamics: control and display latency, pointer gain and rotation,
sag. A policy can only act well if it infers them from how the world responded to its own
recent commands. `bc-history` (learning/policy.py) sees a fixed window of 4 steps; this
policy carries a recurrent state from the start of the episode, so it can integrate evidence
over seconds, as the non-privileged adaptive expert (learning/adaptive.py) does explicitly.

    step t input   low-dimensional features of obs_t, and the action executed at t-1
    network        MLP encoder -> GRU -> MLP head on [encoding, GRU state]
    output         [waypoint offset (cm), finger], as the BC policy (decode_target)

Training: truncated backpropagation through time. Episodes are bucketed by length into
batches; each batch is unrolled in chunks of CHUNK steps, carrying the (detached) state from
chunk to chunk, so the policy is trained, as it runs, on states that start at the episode's
first step. Loss and validation split as BC (smooth L1, finger weighted 4x, episodes held out).

    python main.py train --recurrent --data data/hard_train.npz --seed 0 --out checkpoints/hard_bcr_s0.pt
    python -m benchmark run --agent bc:checkpoints/hard_bcr_s0.pt --all-hard   # agent name bc-gru
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from learning.dataset import FEATURE_DIM, Demonstrations, decode_target, low_dim_features, normalize_action

CHUNK = 128  # steps per truncated-BPTT segment (2.56 s at 50 Hz)


class RecurrentBCNetwork(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM, hidden: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_dim + 4, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        # The head also sees the current encoding directly: the precise part of the action
        # (where the key is now) does not have to pass through the recurrent state.
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4))

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, feature_dim + 4) -> outputs (B, T, 4), final GRU state."""
        e = self.encoder(x)
        h, state = self.gru(e, state)
        y = self.head(torch.cat([e, h], dim=-1))
        return torch.cat([y[..., :3], torch.sigmoid(y[..., 3:])], dim=-1), state


def episode_inputs(demos: Demonstrations) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per episode (in order): inputs (T, FEATURE_DIM + 4) and targets (T, 4). The action
    input at step t is the action executed at t-1 (zeros at the first step)."""
    if demos.actions is None:
        raise ValueError("recurrent BC needs demonstrations with recorded actions; re-collect them")
    if np.any(np.diff(demos.episode) < 0):
        raise ValueError("episodes must be stored contiguously and in order")
    bounds = np.flatnonzero(np.diff(demos.episode)) + 1
    out = []
    for idx in np.split(np.arange(len(demos)), bounds):
        prev = np.zeros((len(idx), 4), np.float32)
        prev[1:] = demos.actions[idx[:-1]]
        out.append((np.concatenate([demos.features[idx], prev], axis=1).astype(np.float32), demos.targets[idx]))
    return out


def _batches(lengths: np.ndarray, size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Episodes of similar length together (little padding), batches in random order."""
    order = np.argsort(lengths + rng.random(len(lengths)))  # random tie-breaks
    batches = [order[i:i + size] for i in range(0, len(order), size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def _pad(episodes: list[tuple[np.ndarray, np.ndarray]], idx: np.ndarray):
    t = max(len(episodes[i][0]) for i in idx)
    x = np.zeros((len(idx), t, episodes[idx[0]][0].shape[1]), np.float32)
    y = np.zeros((len(idx), t, 4), np.float32)
    mask = np.zeros((len(idx), t), np.float32)
    for row, i in enumerate(idx):
        n = len(episodes[i][0])
        x[row, :n], y[row, :n], mask[row, :n] = episodes[i][0], episodes[i][1], 1.0
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


def train_recurrent_bc(demos: Demonstrations, out_path: str | Path, epochs: int = 40, batch_episodes: int = 16,
                       lr: float = 1e-3, hidden: int = 256, seed: int = 0, device: str = "auto",
                       val_fraction: float = 0.1, loss_beta: float = 0.05, verbose: bool = True) -> dict:
    from learning.policy import pick_device

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = pick_device(device)
    episodes = episode_inputs(demos)
    n_val = max(1, int(len(episodes) * val_fraction)) if len(episodes) > 1 else 0
    is_val = np.zeros(len(episodes), bool)
    is_val[rng.choice(len(episodes), n_val, replace=False)] = True
    train_eps, val_eps = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    lengths = np.array([len(e[0]) for e in episodes])

    net = RecurrentBCNetwork(episodes[0][0].shape[1] - 4, hidden).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    weights = torch.tensor([1.0, 1.0, 1.0, 4.0], device=dev)

    def chunk_loss(pred, y, mask):
        err = nn.functional.smooth_l1_loss(pred, y, reduction="none", beta=loss_beta) * weights
        return (err.mean(-1) * mask).sum(), mask.sum()

    @torch.no_grad()
    def validate() -> float:
        net.eval()
        total, count = 0.0, 0.0
        for b in _batches(lengths[val_eps], batch_episodes, np.random.default_rng(0)):
            x, y, mask = (t.to(dev) for t in _pad(episodes, val_eps[b]))
            pred, _ = net(x)
            loss, n = chunk_loss(pred, y, mask)
            total, count = total + loss.item(), count + n.item()
        return total / max(count, 1.0)

    history = {"train": [], "val": []}
    for epoch in range(epochs):
        net.train()
        total, count = 0.0, 0.0
        for b in _batches(lengths[train_eps], batch_episodes, rng):
            x, y, mask = (t.to(dev) for t in _pad(episodes, train_eps[b]))
            state = None
            for s in range(0, x.shape[1], CHUNK):
                pred, state = net(x[:, s:s + CHUNK], state)
                loss, n = chunk_loss(pred, y[:, s:s + CHUNK], mask[:, s:s + CHUNK])
                opt.zero_grad()
                (loss / n.clamp(min=1.0)).backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                state = state.detach()
                total, count = total + loss.item(), count + n.item()
        sched.step()
        history["train"].append(total / max(count, 1.0))
        history["val"].append(validate() if len(val_eps) else float("nan"))
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  train {history['train'][-1]:.5f}  val {history['val'][-1]:.5f}", flush=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "recurrent_bc", "state_dict": net.state_dict(), "feature_dim": episodes[0][0].shape[1] - 4,
                "hidden": hidden, "loss_beta": loss_beta, "chunk": CHUNK, "epochs": epochs, "history": history,
                "num_samples": len(demos), "task_specs": demos.task_specs}, out_path)
    if verbose:
        print(f"saved {out_path} ({len(train_eps)} train / {len(val_eps)} val episodes)")
    return history


class RecurrentBCPolicy:
    """Runs a RecurrentBCNetwork step by step, carrying its state through the episode."""

    name = "bc-gru"

    def __init__(self, checkpoint: str | Path, max_step: float, device: str = "cpu"):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        if ckpt.get("kind") != "recurrent_bc":
            raise ValueError(f"{checkpoint} is not a recurrent BC checkpoint")
        self.device = torch.device(device)
        self.net = RecurrentBCNetwork(ckpt["feature_dim"], ckpt["hidden"]).to(self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.max_step = max_step
        self.skill = "learned"
        self.reset()

    def reset(self) -> None:
        self.state = None
        self.prev_action = np.zeros(4, np.float32)

    @torch.no_grad()
    def act(self, obs: dict) -> np.ndarray:
        x = np.concatenate([low_dim_features(obs), self.prev_action]).astype(np.float32)
        pred, self.state = self.net(torch.from_numpy(x).view(1, 1, -1).to(self.device), self.state)
        action = decode_target(pred[0, 0].cpu().numpy(), self.max_step)
        self.prev_action = normalize_action(action, self.max_step)
        return action
