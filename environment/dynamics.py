"""Hidden real-world dynamics: the "realistic" physics profile.

Scene variation (environment/variation.py) changes what the world *looks* like and where
things are. Hidden dynamics change how the world *responds*, and are never observed: an agent
has to infer them from the consequences of its own actions, within the episode. Each
parameter models an effect that real robots and real computers have:

    action_delay      control latency: commands take effect 0-4 steps (0-80 ms) late
    action_noise      actuation noise on every hand-target change
    obs_noise         noise on proprioception and on the state-track positions
    display_latency   the monitor (and the cursor seen on it) lags the computer by N frames
    gravcomp          imperfect gravity compensation: the arm sags under its own weight
    pointer_accel     OS pointer acceleration: the cursor moves further for fast mouse motion
    mouse_yaw         the mouse starts rotated; its optical sensor reports motion in the
                      mouse's own frame, so a rotated mouse moves the cursor diagonally, and
                      dragging can rotate it further (it is a free planar body on the pad)
    key_stiffness     every key has its own spring force (switch-to-switch variation)
    key_actuation     every key has its own actuation depth (fraction of its 4 mm travel)
    autorepeat_delay  holding a key longer than this repeats the character (30 Hz)
    double_click_time files open only on a double-click within this window; a single click
                      just selects

Distributions:

    none       nominal v1.0 physics (no hidden dynamics)
    real_id    the training range of every parameter
    real_ood   every parameter beyond its training range (longer latency, stronger
               acceleration, more sag, faster auto-repeat, tighter double-click, ...)

The realistic profile needs the mouse's rotation joint, so an environment is built either
for nominal physics (exactly v1.0) or for the realistic profile (EnvConfig.physics).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

DYNAMICS = ("none", "real_id", "real_ood")


@dataclass(frozen=True)
class HiddenDynamics:
    action_delay: int = 0  # control steps
    action_noise: float = 0.0  # m (std) added to each hand-target change
    obs_noise: float = 0.0  # m (std) on observed positions; joints get obs_noise * 5 rad
    display_latency: int = 0  # control steps
    gravcomp: float = 1.0  # fraction of the arm's weight the controller cancels
    pointer_accel: float = 0.0  # extra gain at high mouse speed (0 = linear mouse)
    mouse_yaw: float = 0.0  # initial mouse rotation, rad
    key_stiffness: tuple[float, ...] | None = None  # per-key spring multipliers
    key_actuation: tuple[float, ...] | None = None  # per-key actuation point, fraction of travel
    autorepeat_delay: float = float("inf")  # s
    double_click_time: float | None = None  # s; None: a single click opens a file
    seed: int = 0  # seeds the action / observation noise of the episode
    name: str = "none"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["autorepeat_delay"] = None if np.isinf(self.autorepeat_delay) else self.autorepeat_delay
        return d


NOMINAL_DYNAMICS = HiddenDynamics()


def sample_dynamics(distribution: str, rng: np.random.Generator, n_keys: int) -> HiddenDynamics:
    if distribution == "none":
        return NOMINAL_DYNAMICS
    if distribution not in DYNAMICS:
        raise ValueError(f"unknown dynamics {distribution!r}; choose from {DYNAMICS}")
    u = lambda lo, hi: float(rng.uniform(lo, hi))
    sign = lambda: float(rng.choice([-1.0, 1.0]))
    if distribution == "real_id":
        return HiddenDynamics(
            action_delay=int(rng.integers(0, 2)),  # 0-20 ms
            action_noise=u(0.0, 0.0002),
            obs_noise=u(0.0, 0.0003),
            display_latency=int(rng.integers(0, 3)),  # 0-40 ms
            gravcomp=u(0.95, 1.0),
            pointer_accel=u(0.0, 0.5),
            mouse_yaw=u(-0.15, 0.15),  # +-8.6 deg
            key_stiffness=tuple(rng.uniform(0.8, 1.4, n_keys).round(3).tolist()),
            key_actuation=tuple(rng.uniform(0.5, 0.7, n_keys).round(3).tolist()),
            autorepeat_delay=u(0.45, 0.6),
            double_click_time=u(0.4, 0.5),
            seed=int(rng.integers(2**31)),
            name=distribution,
        )
    return HiddenDynamics(
        action_delay=int(rng.integers(2, 5)),  # 40-80 ms
        action_noise=u(0.0003, 0.0005),
        obs_noise=u(0.0005, 0.0008),
        display_latency=int(rng.integers(3, 6)),  # 60-100 ms
        gravcomp=u(0.85, 0.9),
        pointer_accel=u(0.8, 1.2),
        mouse_yaw=sign() * u(0.25, 0.4),  # 14-23 deg
        key_stiffness=tuple(rng.uniform(1.5, 1.9, n_keys).round(3).tolist()),
        key_actuation=tuple(rng.uniform(0.75, 0.85, n_keys).round(3).tolist()),
        autorepeat_delay=u(0.25, 0.35),
        double_click_time=u(0.28, 0.33),
        seed=int(rng.integers(2**31)),
        name=distribution,
    )
