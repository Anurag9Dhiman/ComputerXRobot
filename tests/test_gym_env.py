"""Gymnasium interface."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import environment.gym_env as gym_env
from environment.task_manager import SPLITS
from learning.dataset import FEATURE_DIM


# The state features are unbounded (like Gymnasium's own MuJoCo envs); the checker notes that.
@pytest.mark.filterwarnings("ignore:.*Box observation space m.*:UserWarning")
@pytest.mark.parametrize("obs_mode", ["state", "pixels", "both"])
def test_passes_gymnasium_env_checker(obs_mode):
    env = gym_env.ComputerUseGymEnv(tasks=("type_key",), obs_mode=obs_mode)
    check_env(env, skip_render_check=True)
    env.close()


def test_registered_ids():
    for env_id in gym_env.ENV_IDS:
        assert env_id in gym.registry
    env = gym.make("ComputerXRobot/TypeText-v0")
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["task"] == "type_text" and info["text"] in SPLITS["type_text"]["train"]
    env.close()


def test_expert_solves_search_through_gym_api():
    env = gym.make("ComputerXRobot/Search-v0")
    env.reset(seed=0, options={"task": "search", "text": "mouse control"})
    total, terminated, truncated = 0.0, False, False
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(env.unwrapped.expert_action())
        total += reward
    assert terminated and not truncated and info["is_success"]
    assert info["ui_state"]["browser_last_query"] == "mouse control"
    assert total == pytest.approx(2.0)  # progress 0 -> 1, plus the success bonus
    assert info["metrics"]["wrong_keys"] == info["metrics"]["failed_clicks"] == 0
    env.close()


def test_seeded_task_sampling_is_reproducible():
    env = gym.make("ComputerXRobot/ComputerUse-v0", split="test")
    first = [env.reset(seed=s)[1] for s in range(10)]
    second = [env.reset(seed=s)[1] for s in range(10)]
    assert [(i["task"], i["text"]) for i in first] == [(i["task"], i["text"]) for i in second]
    assert len({i["task"] for i in first}) > 1
    assert all(i["text"] in SPLITS[i["task"]]["test"] for i in first)
    env.close()


def test_idle_robot_is_truncated_at_the_time_limit():
    env = gym.make("ComputerXRobot/TypeKey-v0")
    env.reset(seed=0)
    idle = np.array([0, 0, 0, -1], dtype=np.float32)  # hold still, finger retracted
    steps, terminated, truncated = 0, False, False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(idle)
        steps += 1
    assert truncated and not terminated and not info["is_success"]
    assert abs(steps - env.unwrapped.task.time_limit / 0.02) <= 1


def test_vector_env():
    envs = gym.vector.SyncVectorEnv([lambda: gym.make("ComputerXRobot/TypeKey-v0") for _ in range(2)])
    obs, info = envs.reset(seed=[0, 1])
    assert obs.shape == (2, FEATURE_DIM)
    for _ in range(5):
        obs, reward, terminated, truncated, info = envs.step(envs.action_space.sample())
    assert reward.shape == (2,) and obs.dtype == np.float32
    envs.close()


def test_rgb_array_render_is_the_dashboard():
    env = gym.make("ComputerXRobot/TypeKey-v0", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (720, 1280, 3) and frame.dtype == np.uint8
    env.close()


def test_device_cameras_and_ground_truth_through_gym():
    cams = ("robot_cam", "keyboard_cam", "mouse_cam", "screen_cam")
    env = gym.make("ComputerXRobot/ClearText-v0", obs_mode="pixels", cameras=cams)
    obs, info = env.reset(seed=0)
    assert obs["keyboard_cam"].shape == (256, 512, 3) and obs["mouse_cam"].shape == (320, 320, 3)
    assert obs["screen"].shape == (400, 640, 3)  # native resolution: text stays legible
    assert info["text"] in [t for t, _ in info["screen_text"]]  # the prefilled editor text
    buttons = [r["label"] for r in env.unwrapped.annotations("mouse_cam") if r["kind"] == "mouse_button"]
    assert sorted(buttons) == ["L", "R"]
    env.close()
