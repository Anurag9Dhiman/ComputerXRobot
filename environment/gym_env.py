"""Gymnasium interface to the computer-use simulation.

    import gymnasium as gym
    import environment.gym_env  # registers the ComputerXRobot/* environments

    env = gym.make("ComputerXRobot/Search-v0", obs_mode="pixels")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Registered IDs: ComputerXRobot/ComputerUse-v0 (all tasks), TypeKey-v0, TypeText-v0,
Search-v0, OpenResult-v0, OpenApp-v0. `gym.make("environment.gym_env:ComputerXRobot/...")`
imports this module automatically.

Action: Box(-1, 1, (4,)) = [dx, dy, dz, finger]. dx..dz move the tool target by up to
max_step (6 mm) per step; finger maps [-1, 1] to the finger command [0, 1] (retracted to
fully extended).

Observation, by `obs_mode`:
    "state"   Box(66,) float32: proprioception, device state, cursor, click target and the
              next character to type (learning.dataset.low_dim_features)
    "pixels"  Dict of uint8 images: one entry per camera in `cameras` plus "screen"
    "both"    the pixel Dict plus "state"

Episodes: each reset samples a task from `tasks` and `split` with the env's RNG, or uses
options={"task": name, "text": text}. `terminated` means the task succeeded; `truncated` means
its time limit ran out. The reward is the simulation's shaped reward: progress gained,
+1 on success, -0.1 per wrong key or failed click.

`info` carries ground truth for every step: task, success, progress, the desktop state
(text, cursor, page, ...) and the UI events the step produced. `env.unwrapped.expert_action()`
returns the scripted expert's action for the current state, for demonstrations or mixed
exploration.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.computer_env import ComputerUseEnv, EnvConfig
from environment.task_manager import SPLITS, TASKS, Task, make_task
from learning.dataset import FEATURE_DIM, low_dim_features
from learning.scripted_policy import ScriptedPolicy

OBS_MODES = ("state", "pixels", "both")


class ComputerUseGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}  # one frame per control step

    def __init__(
        self,
        tasks: tuple[str, ...] | list[str] = tuple(TASKS),
        split: str = "train",
        obs_mode: str = "state",
        cameras: tuple[str, ...] | list[str] = ("robot_cam",),
        image_size: tuple[int, int] = (64, 64),
        screen_size: tuple[int, int] = (64, 96),
        render_mode: str | None = None,
    ):
        unknown = set(tasks) - set(TASKS)
        if unknown:
            raise ValueError(f"unknown tasks {sorted(unknown)}; choose from {sorted(TASKS)}")
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        if obs_mode not in OBS_MODES:
            raise ValueError(f"obs_mode must be one of {OBS_MODES}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")

        self.tasks, self.split, self.obs_mode, self.render_mode = tuple(tasks), split, obs_mode, render_mode
        self.cameras = tuple(cameras)
        self.sim = ComputerUseEnv(
            EnvConfig(
                render_obs=obs_mode != "state",
                obs_cameras=self.cameras,
                obs_image_size=tuple(image_size),
                obs_screen_size=tuple(screen_size),
            )
        )
        self._max_step = self.sim.config.max_step
        self._expert = ScriptedPolicy(self.sim)
        self._dashboard = None
        self._window = self._clock = None

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        state_space = spaces.Box(-np.inf, np.inf, shape=(FEATURE_DIM,), dtype=np.float32)
        if obs_mode == "state":
            self.observation_space = state_space
        else:
            images = {cam: spaces.Box(0, 255, (*image_size, 3), np.uint8) for cam in self.cameras}
            images["screen"] = spaces.Box(0, 255, (*screen_size, 3), np.uint8)
            if obs_mode == "both":
                images["state"] = state_space
            self.observation_space = spaces.Dict(images)

    # ---------------------------------------------------------------- core API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        obs = self.sim.reset(self._choose_task(options or {}))
        self._expert.reset()
        if self.render_mode == "human":
            self.render()
        return self._convert(obs), self._info([])

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        sim_action = np.array([*(a[:3] * self._max_step), (a[3] + 1.0) / 2.0])
        obs, reward, _, step_info = self.sim.step(sim_action)
        status = step_info.status
        terminated, truncated = bool(status.success), bool(status.timeout)
        info = self._info(step_info.events)
        if terminated or truncated:
            info["metrics"] = self.sim.metrics()
        if self.render_mode == "human":
            self.render()
        return self._convert(obs), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("render() called without a render_mode; pass render_mode='rgb_array' or 'human'")
            return None
        import pygame

        from evaluation.dashboard import HEIGHT, WIDTH, Dashboard

        if self._dashboard is None:
            self._dashboard = Dashboard(self.sim)
        frame = self._dashboard.draw()
        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(frame).transpose(1, 0, 2).copy()
        if self._window is None:
            pygame.display.init()
            self._window = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("ComputerXRobot")
            self._clock = pygame.time.Clock()
        pygame.event.pump()
        self._window.blit(frame, (0, 0))
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
        return None

    def close(self):
        if self._window is not None:
            import pygame

            pygame.display.quit()
            self._window = None
        self.sim.close()

    # ----------------------------------------------------------------- helpers
    def expert_action(self) -> np.ndarray:
        """The scripted expert's action for the current state, in this env's action space."""
        a = self._expert.act()
        normalized = np.array([*(a[:3] / self._max_step), 2.0 * a[3] - 1.0], dtype=np.float32)
        return np.clip(normalized, -1.0, 1.0)

    @property
    def task(self) -> Task:
        return self.sim.task

    def _choose_task(self, options: dict[str, Any]) -> Task:
        task = options.get("task")
        if isinstance(task, Task):
            return task
        if task is not None:
            return make_task(task, options.get("text"))
        name = self.tasks[int(self.np_random.integers(len(self.tasks)))]
        texts = SPLITS[name][self.split]
        return make_task(name, texts[int(self.np_random.integers(len(texts)))])

    def _convert(self, obs: dict):
        if self.obs_mode == "state":
            return low_dim_features(obs)
        out = {cam: np.ascontiguousarray(obs[cam]) for cam in self.cameras}
        out["screen"] = np.ascontiguousarray(obs["screen"])
        if self.obs_mode == "both":
            out["state"] = low_dim_features(obs)
        return out

    def _info(self, events) -> dict[str, Any]:
        task, status = self.sim.task, self.sim.status
        return {
            "task": task.name,
            "text": getattr(task, "text", None),
            "instruction": task.instruction,
            "is_success": bool(status.success),
            "progress": float(status.progress),
            "expected": self.sim.tasks.expectation(),
            "ui_state": self.sim.desktop.state(),
            "events": [{"kind": e.kind, "value": e.value, "correct": e.correct} for e in events],
        }


ENV_IDS = {
    "ComputerXRobot/ComputerUse-v0": {},
    "ComputerXRobot/TypeKey-v0": {"tasks": ("type_key",)},
    "ComputerXRobot/TypeText-v0": {"tasks": ("type_text",)},
    "ComputerXRobot/Search-v0": {"tasks": ("search",)},
    "ComputerXRobot/OpenResult-v0": {"tasks": ("open_result",)},
    "ComputerXRobot/OpenApp-v0": {"tasks": ("open_app",)},
}


def register_envs() -> None:
    for env_id, kwargs in ENV_IDS.items():
        if env_id not in gym.registry:
            gym.register(id=env_id, entry_point="environment.gym_env:ComputerUseGymEnv", kwargs=kwargs)


register_envs()
