"""Dynamics-prediction track: windowing and scoring are correct on synthetic data with a
known answer."""

import numpy as np
import pytest

from benchmark.world_model import (CONTEXT, CURSOR, HORIZON, OBS_DIM, PALM, TOOL, Kinematic, Persistence, score,
                                   windows)


def _episode(T, on_mouse, gain=6400.0, max_step=0.006, seed=0):
    """A synthetic episode whose physics is exactly the kinematic model."""
    rng = np.random.default_rng(seed)
    actions = np.zeros((T, 4), np.float32)
    actions[:, :2] = rng.uniform(-1, 1, (T, 2))
    obs = np.zeros((T, OBS_DIM), np.float32)
    target = np.zeros(3)
    cursor = np.array([320.0, 200.0])
    for t in range(T):
        obs[t, TOOL] = target  # instant tracking
        obs[t, 3:6] = target
        obs[t, CURSOR] = cursor
        obs[t, PALM] = float(on_mouse)
        step = actions[t, :3] * max_step
        target = target + step
        if on_mouse:
            cursor = cursor + gain * np.array([-step[1], -step[0]])
    return obs, actions


def _data(*episodes):
    obs = np.concatenate([e[0] for e in episodes])
    act = np.concatenate([e[1] for e in episodes])
    starts = np.cumsum([0] + [len(e[0]) for e in episodes])
    return {"obs": obs, "actions": act, "starts": starts, "dynamics": [{}] * len(episodes)}


def test_windows_stay_inside_episodes():
    data = _data(_episode(60, True), _episode(45, False, seed=1))
    co, ca, fa, fo = windows(data)
    assert co.shape[1:] == (CONTEXT, OBS_DIM) and fa.shape[1:] == (HORIZON, 4) and fo.shape[1:] == (HORIZON, OBS_DIM)
    # current step t = 20, 30 in the 60-step episode and t = 20 in the 45-step one
    assert len(co) == 2 + 1
    assert np.allclose(co[0], data["obs"][1:21]) and np.allclose(fo[0], data["obs"][21:41])
    assert np.allclose(fa[0], data["actions"][20:40])


def test_kinematic_model_is_exact_on_kinematic_physics_and_persistence_is_not():
    data = _data(_episode(120, True), _episode(120, True, seed=2))
    exact = score(Kinematic(), data)
    lazy = score(Persistence(), data)
    assert exact["cursor_px@20"] == pytest.approx(0.0, abs=1e-3) and exact["hand_mm@20"] == pytest.approx(0.0, abs=1e-3)
    assert lazy["cursor_px@20"] > 50 and lazy["hand_mm@20"] > 5
