"""Demonstrations: feature extraction, collection with the scripted expert, storage.

Low-dimensional policy input (`low_dim_features`, FEATURE_DIM values):
    tool target / tracking error / finger extension      proprioception
    mouse offset, palm-on-mouse, key/button down          device state
    cursor, click target, target - cursor                 what the screen shows
    mouse button relative to the tool                     where the mouse is
    one-hot of the next character to type                 task state

Labels ("targets") are the expert's intent rather than its clipped step: the offset from
the current tool target to the expert's waypoint (in cm, length clipped to 5 cm) plus the
finger command. The waypoint offset is close to a linear function of the features (key
position minus tool position, cursor error divided by the mouse gain), which an MLP fits far
more precisely than the saturated per-step action. `decode_target` turns a prediction back
into an environment action exactly the way the expert does.

Collection injects perturbations in the spirit of DART: every so often the executed motion
is pushed off course for a few steps (scaled by the expert's per-phase `noise_scale`), while
the recorded label stays the clean expert action. The expert then recovers, so the data
covers the off-nominal states a slightly imperfect learned policy will visit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from environment.browser import SCREEN_H, SCREEN_W
from environment.computer_env import ComputerUseEnv
from environment.keyboard_layout import KEYS
from environment.task_manager import Task
from learning.scripted_policy import ScriptedPolicy, waypoint_action

WORKSPACE_CENTER = np.array([0.51, -0.075, 0.88])
WORKSPACE_SCALE = 0.2
CHAR_INDEX = {k.char: i for i, k in enumerate(KEYS)}
FEATURE_DIM = 3 + 3 + 1 + 2 + 3 + 2 + 1 + 2 + 2 + 3 + len(KEYS)
IMAGE_KEYS = ("robot_cam", "screen")
WAYPOINT_SCALE = 0.01  # label units: 1.0 = 1 cm
WAYPOINT_CLIP = 0.05  # m; longer offsets only matter through their direction


def _screen_norm(px) -> np.ndarray:
    return np.asarray(px, float) / [SCREEN_W / 2, SCREEN_H / 2] - 1.0


def low_dim_features(obs: dict) -> np.ndarray:
    f = np.zeros(FEATURE_DIM, dtype=np.float32)
    i = 0

    def put(values):
        nonlocal i
        values = np.atleast_1d(np.asarray(values, dtype=np.float32))
        f[i : i + len(values)] = values
        i += len(values)

    put((obs["tool_target"] - WORKSPACE_CENTER) / WORKSPACE_SCALE)
    put((obs["tool_pos"] - obs["tool_target"]) * 100.0)
    put(obs["finger"])
    put(obs["mouse_pos"] / 0.08)
    put([obs["palm_on_mouse"], obs["key_down"], obs["button_down"]])
    put(_screen_norm(obs["cursor"]))
    target = obs["click_target_px"]
    put(float(target is not None))
    if target is not None:
        put(_screen_norm(target))
        put((np.asarray(target, float) - obs["cursor"]) / 50.0)
    else:
        i += 4
    put((obs["mouse_button"] - obs["tool_pos"]) * 10.0)
    if obs["expected_char"] is not None:
        f[i + CHAR_INDEX[obs["expected_char"]]] = 1.0
    i += len(KEYS)
    assert i == FEATURE_DIM
    return f


def encode_target(waypoint_offset: np.ndarray, finger: float) -> np.ndarray:
    offset = np.asarray(waypoint_offset, float)
    dist = np.linalg.norm(offset)
    if dist > WAYPOINT_CLIP:
        offset = offset * (WAYPOINT_CLIP / dist)
    return np.array([*(offset / WAYPOINT_SCALE), finger], dtype=np.float32)


def decode_target(target: np.ndarray, max_step: float) -> np.ndarray:
    """Model output [offset (cm), finger] -> environment action."""
    return waypoint_action(np.asarray(target[:3], float) * WAYPOINT_SCALE, float(target[3]), max_step)


@dataclass
class Demonstrations:
    features: np.ndarray  # (N, FEATURE_DIM) float32
    targets: np.ndarray  # (N, 4) float32: waypoint offset (cm), finger command
    episode: np.ndarray  # (N,) int32
    images: dict[str, np.ndarray]  # name -> (N, H, W, 3) uint8 (may be empty)
    task_specs: list[dict]  # one per episode

    def __len__(self) -> int:
        return len(self.targets)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features,
            targets=self.targets,
            episode=self.episode,
            task_specs=np.array([json.dumps(s) for s in self.task_specs]),
            **{f"image_{k}": v for k, v in self.images.items()},
        )

    @classmethod
    def load(cls, path: str | Path) -> "Demonstrations":
        z = np.load(path)
        images = {k[len("image_"):]: z[k] for k in z.files if k.startswith("image_")}
        specs = [json.loads(s) for s in z["task_specs"]]
        return cls(z["features"], z["targets"], z["episode"], images, specs)

    @classmethod
    def concatenate(cls, parts: list["Demonstrations"]) -> "Demonstrations":
        offset, episodes = 0, []
        for p in parts:
            episodes.append(p.episode + offset)
            offset += len(p.task_specs)
        keys = set.intersection(*(set(p.images) for p in parts)) if parts else set()
        return cls(
            np.concatenate([p.features for p in parts]),
            np.concatenate([p.targets for p in parts]),
            np.concatenate(episodes),
            {k: np.concatenate([p.images[k] for p in parts]) for k in keys},
            [s for p in parts for s in p.task_specs],
        )


def collect_demonstrations(
    env: ComputerUseEnv,
    tasks: list[Task],
    noise: float = 0.5,
    perturb_prob: float = 0.03,
    seed: int = 0,
    with_images: bool = False,
    keep_failures: bool = False,
    verbose: bool = True,
) -> Demonstrations:
    """Roll out the scripted expert on `tasks` and record (features, expert target) pairs.

    noise         perturbation size as a fraction of the max step per control step
    perturb_prob  per-step probability of starting a 3-9 step perturbation
    """
    rng = np.random.default_rng(seed)
    expert = ScriptedPolicy(env)
    max_step = env.config.max_step
    feats, targets, eps, specs = [], [], [], []
    images: dict[str, list] = {k: [] for k in IMAGE_KEYS} if with_images else {}
    for task in tasks:
        obs = env.reset(task)
        expert.reset()
        ep_feats, ep_targets, ep_imgs = [], [], {k: [] for k in images}
        done = False
        push, push_steps = np.zeros(3), 0
        while not done:
            action = expert.act(obs)
            ep_feats.append(low_dim_features(obs))
            ep_targets.append(encode_target(expert.waypoint - env.tool_target, expert.finger))
            for k in images:
                ep_imgs[k].append(obs[k])
            if push_steps == 0 and rng.random() < perturb_prob:
                direction = rng.normal(size=3)
                push, push_steps = direction / np.linalg.norm(direction) * noise * max_step, int(rng.integers(3, 10))
            executed = action.copy()
            if push_steps:
                executed[:3] += push * expert.noise_scale
                push_steps -= 1
            obs, _, done, _ = env.step(executed)
        ok = env.status.success
        if verbose:
            print(f"  {task.name:12s} {getattr(task, 'text', '')!r:32s} "
                  f"{'ok ' if ok else 'FAIL'} {len(ep_targets):4d} steps")
        if ok or keep_failures:
            feats += ep_feats
            targets += ep_targets
            eps += [len(specs)] * len(ep_targets)
            specs.append(task.spec())
            for k in images:
                images[k] += ep_imgs[k]
    return Demonstrations(
        np.asarray(feats, np.float32).reshape(-1, FEATURE_DIM),
        np.asarray(targets, np.float32).reshape(-1, 4),
        np.asarray(eps, np.int32),
        {k: np.asarray(v, np.uint8) for k, v in images.items()},
        specs,
    )
