"""Dynamics-prediction track: does a model know how the world will respond?

Under the hidden dynamics of the hard tier (environment/dynamics.py) the effect of an action
depends on parameters nobody tells the model: control and display latency, pointer
acceleration, the mouse's rotation, friction, arm sag. This track scores world models
directly on predicting those effects.

Task. Given C = 20 control steps (0.4 s) of observations and commanded actions, and the next
H = 20 commanded actions, predict the next H observations of
    the cursor position on screen (px), and
    the hand (tool) position (mm).
Observations are what an agent sees: with display latency the observed cursor lags, and
observations are noisy. A good model must therefore infer the episode's dynamics from the
context and roll them forward.

Data. Trajectories of the oracle with DART-style perturbations under hidden dynamics:
`hard_train` episodes to train on; `hard_test` (training ranges) and `hard_test_ood` (beyond
them) to evaluate. Windows start every 10 steps.

Metrics (lower is better), at horizons 1, 5, 10, 20 steps (20-400 ms):
    cursor error   mean distance (px), over windows in which the hand holds the mouse
    hand error     mean distance (mm), over all windows

Baselines:
    persistence    nothing moves
    kinematic      a physics-free prior: the hand reaches each commanded target instantly and
                   the cursor moves CURSOR_GAIN px/m with the hand while the palm is on the
                   mouse (no tracking lag, latency, acceleration, rotation or sag)
    mlp            an MLP trained on hard_train windows: context + future actions -> future
                   displacements

    python -m benchmark.world_model collect        # trajectories -> data/world_model_*.npz
    python -m benchmark.world_model train          # MLP baseline -> checkpoints/world_model.pt
    python -m benchmark.world_model evaluate       # table for all baselines
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CONTEXT, HORIZON, STRIDE = 20, 20, 10
HORIZONS = (1, 5, 10, 20)
# Observation vector per step (what an agent observes; cursor lags under display latency).
OBS_FIELDS = ("tool_pos", "tool_target", "finger", "mouse_pos", "cursor", "palm_on_mouse", "key_down", "button_down")
OBS_DIM = 3 + 3 + 1 + 2 + 2 + 1 + 1 + 1
TOOL, CURSOR, PALM = slice(0, 3), slice(9, 11), 11
SPLIT_FILES = {"hard_train": "world_model_train.npz", "hard_test": "world_model_test.npz",
               "hard_test_ood": "world_model_test_ood.npz"}


def obs_vector(obs: dict) -> np.ndarray:
    return np.concatenate([np.atleast_1d(np.asarray(obs[k], float)) for k in OBS_FIELDS]).astype(np.float32)


# ------------------------------------------------------------------------------- data
def collect(split: str, per_task: int, seed: int = 0, noise: float = 0.5, perturb_prob: float = 0.05) -> dict:
    """Roll out the perturbed oracle on `split` and record observation / action sequences."""
    import environment  # noqa: F401
    from benchmark.suite import episodes
    from environment.computer_env import ComputerUseEnv, EnvConfig
    from learning.dataset import normalize_action
    from learning.scripted_policy import ScriptedPolicy

    env = ComputerUseEnv(EnvConfig(render_obs=False, physics="realistic"))
    expert = ScriptedPolicy(env)
    rng = np.random.default_rng(seed)
    obs_all, act_all, starts, dyn = [], [], [0], []
    for spec in episodes(split, per_task=per_task):
        dynamics = spec.hidden_dynamics()
        obs = env.reset(spec.make_task(), variation=spec.variation(), dynamics=dynamics)
        expert.reset()
        push, push_steps, done = np.zeros(3), 0, False
        while not done:
            action = expert.act(obs)
            if push_steps == 0 and rng.random() < perturb_prob:
                d = rng.normal(size=3)
                push, push_steps = d / np.linalg.norm(d) * noise * env.config.max_step, int(rng.integers(3, 10))
            if push_steps:
                action[:3] += push * expert.noise_scale
                push_steps -= 1
            obs_all.append(obs_vector(obs))
            act_all.append(normalize_action(action, env.config.max_step))
            obs, _, done, _ = env.step(action)
        starts.append(len(obs_all))
        dyn.append({**dynamics.to_dict(), "cursor_gain": env.variation.cursor_gain, "episode_id": spec.episode_id})
    return {"obs": np.asarray(obs_all, np.float32), "actions": np.asarray(act_all, np.float32),
            "starts": np.asarray(starts), "dynamics": dyn}


def save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, obs=data["obs"], actions=data["actions"], starts=data["starts"],
                        dynamics=np.array([json.dumps(d) for d in data["dynamics"]]))


def load(path: Path) -> dict:
    z = np.load(path)
    return {"obs": z["obs"], "actions": z["actions"], "starts": z["starts"],
            "dynamics": [json.loads(d) for d in z["dynamics"]]}


def windows(data: dict, stride: int = STRIDE):
    """(context_obs (N,C,D), context_actions (N,C,4), future_actions (N,H,4), future_obs (N,H,D)).

    With t the current step: the context is observations t-C+1..t and the actions t-C..t-1
    that led to them; the model gets actions t..t+H-1 and must predict observations
    t+1..t+H, i.e. future_obs[h-1] is the result of the first h future actions."""
    co, ca, fa, fo = [], [], [], []
    obs, act, starts = data["obs"], data["actions"], data["starts"]
    for a, b in zip(starts[:-1], starts[1:]):
        for t in range(a + CONTEXT, b - HORIZON, stride):
            co.append(obs[t - CONTEXT + 1:t + 1])
            ca.append(act[t - CONTEXT:t])
            fa.append(act[t:t + HORIZON])
            fo.append(obs[t + 1:t + HORIZON + 1])
    return np.asarray(co), np.asarray(ca), np.asarray(fa), np.asarray(fo)


# ----------------------------------------------------------------------- predictors
class Persistence:
    name = "persistence"

    def predict(self, co, ca, fa):
        return np.repeat(co[:, -1:, :], HORIZON, axis=1)


class Kinematic:
    """Physics-free prior: instant tracking, no latency, a linear unrotated mouse."""

    name = "kinematic"

    def __init__(self, max_step: float = 0.006, gain: float = 6400.0):
        self.max_step, self.gain = max_step, gain

    def predict(self, co, ca, fa):
        last = co[:, -1, :]
        steps = np.clip(fa[:, :, :3], -1, 1) * self.max_step  # (N,H,3)
        target = last[:, None, 3:6] + np.cumsum(steps, axis=1)
        pred = np.repeat(last[:, None, :], HORIZON, axis=1).copy()
        pred[:, :, TOOL] = target
        on_mouse = last[:, PALM] > 0.5
        hand = target - last[:, None, TOOL]
        du, dv = -self.gain * hand[:, :, 1], -self.gain * hand[:, :, 0]
        pred[:, :, CURSOR] = last[:, None, CURSOR] + np.where(on_mouse[:, None, None], np.stack([du, dv], -1), 0.0)
        return pred


class MLPWorldModel:
    """context (C x (D+4)) + future actions (H x 4) -> future displacements of tool and cursor."""

    name = "mlp"

    def __init__(self, path: str | Path):
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.net = _make_net(ckpt["in_dim"], ckpt["hidden"])
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.mu, self.sd = ckpt["mu"], ckpt["sd"]

    def predict(self, co, ca, fa):
        import torch

        x = (_inputs(co, ca, fa) - self.mu) / self.sd
        with torch.no_grad():
            d = self.net(torch.from_numpy(x.astype(np.float32))).numpy()
        return _apply_displacements(co, d)


def _relative_context(co):
    """Context observations relative to the last one (positions as displacements)."""
    rel = co.copy()
    rel[:, :, TOOL] -= co[:, -1:, TOOL]
    rel[:, :, 3:6] -= co[:, -1:, TOOL]
    rel[:, :, CURSOR] = (co[:, :, CURSOR] - co[:, -1:, CURSOR]) / 100.0
    rel[:, :, 7:9] -= co[:, -1:, 7:9]
    return rel


def _inputs(co, ca, fa):
    n = len(co)
    last = co[:, -1, :].copy()
    last[:, CURSOR] /= 640.0
    return np.concatenate([_relative_context(co).reshape(n, -1), ca.reshape(n, -1), fa.reshape(n, -1),
                           last], axis=1)


def _targets(co, fo):
    """Future displacements: tool in cm, cursor in units of 100 px."""
    return np.concatenate([(fo[:, :, TOOL] - co[:, -1:, TOOL]) * 100.0,
                           (fo[:, :, CURSOR] - co[:, -1:, CURSOR]) / 100.0], axis=2).reshape(len(co), -1)


def _apply_displacements(co, d):
    d = d.reshape(len(co), HORIZON, 5)
    pred = np.repeat(co[:, -1:, :], HORIZON, axis=1).copy()
    pred[:, :, TOOL] = co[:, -1:, TOOL] + d[:, :, :3] / 100.0
    pred[:, :, CURSOR] = co[:, -1:, CURSOR] + d[:, :, 3:] * 100.0
    return pred


def _make_net(in_dim: int, hidden: int):
    from torch import nn

    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, HORIZON * 5))


def train_mlp(train: dict, out: Path, epochs: int = 60, hidden: int = 512, seed: int = 0, stride: int = 2) -> None:
    import torch

    torch.manual_seed(seed)
    co, ca, fa, fo = windows(train, stride=stride)
    x, y = _inputs(co, ca, fa), _targets(co, fo)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    x = torch.from_numpy(((x - mu) / sd).astype(np.float32))
    y = torch.from_numpy(y.astype(np.float32))
    net = _make_net(x.shape[1], hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        perm = rng.permutation(len(x))
        total = 0.0
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            loss = torch.nn.functional.smooth_l1_loss(net(x[b]), y[b], beta=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  loss {total / len(x):.5f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "in_dim": x.shape[1], "hidden": hidden, "mu": mu, "sd": sd,
                "windows": len(x)}, out)
    print(f"saved {out} ({len(x)} training windows)")


# -------------------------------------------------------------------------- metrics
def score(model, data: dict) -> dict:
    co, ca, fa, fo = windows(data)
    pred = model.predict(co, ca, fa)
    mouse = co[:, -1, PALM] > 0.5  # windows that start with the hand on the mouse
    out = {"windows": int(len(co)), "mouse_windows": int(mouse.sum())}
    for h in HORIZONS:
        cur = np.linalg.norm(pred[:, h - 1, CURSOR] - fo[:, h - 1, CURSOR], axis=1)
        tool = np.linalg.norm(pred[:, h - 1, TOOL] - fo[:, h - 1, TOOL], axis=1) * 1000.0
        out[f"cursor_px@{h}"] = float(cur[mouse].mean())
        out[f"hand_mm@{h}"] = float(tool.mean())
    return out


def table(rows: dict[tuple[str, str], dict]) -> str:
    cols = [f"cursor_px@{h}" for h in HORIZONS] + [f"hand_mm@{h}" for h in HORIZONS]
    head = ["Model", "Split", *[c.replace("_px@", " px @").replace("_mm@", " mm @") for c in cols]]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (model, split), r in rows.items():
        lines.append("| " + " | ".join([model, split, *[f"{r[c]:.1f}" for c in cols]]) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m benchmark.world_model", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--train-per-task", type=int, default=40)
    c.add_argument("--test-per-task", type=int, default=10)
    c.add_argument("--data-dir", default="data")
    t = sub.add_parser("train")
    t.add_argument("--data-dir", default="data")
    t.add_argument("--out", default="checkpoints/world_model.pt")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("evaluate")
    e.add_argument("--data-dir", default="data")
    e.add_argument("--checkpoint", nargs="*", default=["checkpoints/world_model.pt"])
    e.add_argument("--out", help="write the results as JSON")
    args = p.parse_args(argv)

    if args.cmd == "collect":
        for split, name in SPLIT_FILES.items():
            t0 = time.time()
            n = args.train_per_task if split == "hard_train" else args.test_per_task
            data = collect(split, n)
            save(data, Path(args.data_dir) / name)
            print(f"{split}: {len(data['starts']) - 1} episodes, {len(data['obs'])} steps ({time.time() - t0:.0f}s)")
    elif args.cmd == "train":
        train_mlp(load(Path(args.data_dir) / SPLIT_FILES["hard_train"]), Path(args.out), args.epochs, seed=args.seed)
    else:
        models = [Persistence(), Kinematic()] + [MLPWorldModel(c) for c in args.checkpoint if Path(c).exists()]
        rows = {}
        for split in ("hard_test", "hard_test_ood"):
            data = load(Path(args.data_dir) / SPLIT_FILES[split])
            for i, m in enumerate(models):
                label = m.name if m.name != "mlp" else f"mlp (seed {i - 2})"
                rows[(label, split)] = score(m, data)
        print(table(rows))
        if args.out:
            from benchmark.harness import metadata

            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            payload = {"track": "dynamics-prediction", "context": CONTEXT, "horizon": HORIZON,
                       "results": {f"{k[0]}|{k[1]}": v for k, v in rows.items()}, "meta": metadata(1)}
            Path(args.out).write_text(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
