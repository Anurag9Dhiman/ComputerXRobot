"""End-to-end: scripted expert on every task, rendering, and the learning pipeline."""

import numpy as np
import pytest

from environment.computer_env import ComputerUseEnv, EnvConfig
from environment.task_manager import make_task
from evaluation.task_success import run_episode
from learning.dataset import FEATURE_DIM, Demonstrations, collect_demonstrations
from learning.scripted_policy import ScriptedPolicy


@pytest.mark.parametrize(
    "task, text",
    [
        ("type_text", "hello world"),
        ("search", "robotics research papers"),
        ("open_result", "mouse control"),
        ("open_app", "files"),
    ],
)
def test_scripted_expert_solves_task_cleanly(env, task, text):
    m = run_episode(env, ScriptedPolicy(env), make_task(task, text))
    assert m["success"], m
    assert m["wrong_keys"] == m["failed_clicks"] == m["collisions"] == 0, m


def test_rendering_shows_desktop_on_monitor():
    env = ComputerUseEnv(EnvConfig(render_obs=True, obs_image_size=(64, 64), obs_screen_size=(50, 80)))
    try:
        obs = env.reset(make_task("search", "robotics"))
        assert obs["robot_cam"].shape == (64, 64, 3)
        assert obs["screen"].shape == (50, 80, 3)
        before = env.render("screen_cam", 120, 192).astype(int)
        env.desktop.open_app("editor")  # change what the monitor shows
        env.step(np.zeros(4))
        after = env.render("screen_cam", 120, 192).astype(int)
        assert np.abs(after - before).mean() > 2.0  # monitor texture was updated on the GPU
    finally:
        env.close()


def test_collect_save_load_roundtrip(env, tmp_path):
    tasks = [make_task("type_key", "z"), make_task("open_app", "editor")]
    demos = collect_demonstrations(env, tasks, noise=0.5, seed=1, verbose=False)
    assert len(demos.task_specs) == 2
    assert demos.features.shape == (len(demos), FEATURE_DIM)
    assert np.isfinite(demos.features).all() and np.isfinite(demos.targets).all()
    demos.save(tmp_path / "d.npz")
    loaded = Demonstrations.load(tmp_path / "d.npz")
    assert np.array_equal(loaded.targets, demos.targets)
    assert loaded.task_specs == demos.task_specs


def test_bc_training_runs(env, tmp_path):
    torch = pytest.importorskip("torch")
    from learning.policy import BCPolicy, train_bc

    demos = collect_demonstrations(env, [make_task("type_key", "a"), make_task("type_key", "b")], verbose=False)
    train_bc(demos, tmp_path / "bc.pt", epochs=2, verbose=False)
    policy = BCPolicy(tmp_path / "bc.pt", env.config.max_step, device="cpu")
    action = policy.act(env.reset(make_task("type_key", "a")))
    assert action.shape == (4,) and np.isfinite(action).all()
    assert torch.load(tmp_path / "bc.pt", weights_only=False)["num_samples"] == len(demos)
