"""Gym-style environment: a MuJoCo robot operating a simulated computer.

Each `step(action)`:
  1. applies the action through the IK controller,
  2. advances physics for one control period, polling the keyboard/mouse every physics step,
  3. turns physical presses/motion into desktop events and redraws the monitor texture,
  4. checks the task and returns (observation, reward, done, info).

Action (4,): [dx, dy, dz, finger]
    dx, dy, dz  change of the tool-site target in metres (clipped to +-max_step per step)
    finger      finger extension command in [0, 1] (0 = retracted)

Observation (dict):
    robot_cam / wrist_cam / ...  RGB images from `obs_cameras` (if render_obs)
    screen                       RGB image of the simulated display (if render_obs)
    qpos, qvel                   arm joints + finger (6,)
    tool_pos, tool_target        (3,)
    mouse_pos (2,) metres, cursor (2,) pixels, key_down, button_down, palm_on_mouse
    mouse_button (3,)            world position of the mouse button's top face
    expected_char, click_target, click_target_px   task state (see TaskManager.features)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import pygame

from control.contact_detection import DeviceMonitor
from control.robot_controller import RobotController
from environment.browser import Desktop
from environment.keyboard_layout import CAP_CENTER_Z, CAP_HALF_HEIGHT, KEY_BY_CHAR
from environment.screen import SceneRenderer, ScreenTexture, paint_keycap_labels
from environment.task_manager import Task, TaskManager, TaskStatus, make_task
from environment.ui_events import PhysicalUIBridge, UIEvent

SCENE_PATH = Path(__file__).resolve().parent / "scene.xml"


@dataclass
class EnvConfig:
    control_dt: float = 0.02  # 50 Hz control
    max_step: float = 0.006  # m per control step (0.3 m/s)
    obs_cameras: tuple[str, ...] = ("robot_cam",)
    obs_image_size: tuple[int, int] = (128, 128)  # (H, W)
    obs_screen_size: tuple[int, int] = (100, 160)  # (H, W)
    render_obs: bool = True  # False -> no images in observations (fast data collection)
    home_tool_pos: tuple[float, float, float] = (0.47, 0.06, 0.82)
    workspace_low: tuple[float, float, float] = (0.30, -0.45, 0.765)
    workspace_high: tuple[float, float, float] = (0.72, 0.30, 1.00)


@dataclass
class StepInfo:
    status: TaskStatus
    events: list[UIEvent] = field(default_factory=list)


class ComputerUseEnv:
    action_dim = 4

    def __init__(self, config: EnvConfig | None = None, task: Task | None = None):
        self.config = config or EnvConfig()
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.n_substeps = max(1, round(self.config.control_dt / self.model.opt.timestep))

        self.controller = RobotController(self.model)
        self.devices = DeviceMonitor(self.model)
        self.desktop = Desktop()
        self.bridge = PhysicalUIBridge(self.devices, self.desktop)
        self.tasks = TaskManager(self.desktop)
        self.screen = ScreenTexture(self.model)
        paint_keycap_labels(self.model)
        self._renderers: dict[tuple[int, int], SceneRenderer] = {}
        self._screen_version = -1

        self.task = task or make_task("type_key", "a")
        self._low = np.array(self.config.workspace_low)
        self._high = np.array(self.config.workspace_high)
        self._button_site = self.model.site("mouse_button_top").id
        self._rest_site = self.model.site("mouse_rest_top").id
        self._key_tops = self._compute_key_tops()
        self.tool_target = np.array(self.config.home_tool_pos, dtype=float)
        self.finger_cmd = 0.0

    # ------------------------------------------------------------- geometry
    def _compute_key_tops(self) -> dict[str, np.ndarray]:
        d = mujoco.MjData(self.model)
        mujoco.mj_kinematics(self.model, d)
        top = CAP_CENTER_Z + CAP_HALF_HEIGHT
        return {c: d.xpos[self.model.body(k.body).id] + [0, 0, top] for c, k in KEY_BY_CHAR.items()}

    def key_top(self, char: str) -> np.ndarray:
        """World position of the centre of a keycap's top face (key at rest)."""
        return self._key_tops[char].copy()

    def mouse_button_top(self) -> np.ndarray:
        return self.data.site_xpos[self._button_site].copy()

    def mouse_rest_top(self) -> np.ndarray:
        return self.data.site_xpos[self._rest_site].copy()

    # ------------------------------------------------------------- core API
    def reset(self, task: Task | None = None) -> dict:
        if task is not None:
            self.task = task
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        home = np.array(self.config.home_tool_pos, dtype=float)
        self.controller.reset(d, home)
        mujoco.mj_forward(m, d)
        self.tool_target = home
        self.finger_cmd = 0.0
        self.tasks.reset(self.task, d.time)
        self.bridge.reset(d)
        self._sync_screen(force=True)
        self.status = self.tasks.evaluate(d.time)
        self.step_count = 0
        return self.observe()

    def step(self, action) -> tuple[dict, float, bool, StepInfo]:
        action = np.asarray(action, dtype=float)
        delta = np.clip(action[:3], -self.config.max_step, self.config.max_step)
        self.tool_target = np.clip(self.tool_target + delta, self._low, self._high)
        self.finger_cmd = float(np.clip(action[3], 0.0, 1.0))
        self.controller.apply(self.data, self.tool_target, self.finger_cmd)

        events: list[UIEvent] = []
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
            events += self.bridge.update(self.data, self.tasks.expectation)
        self.tasks.record(events)
        self._sync_screen()

        self.status = self.tasks.evaluate(self.data.time)
        self.step_count += 1
        return self.observe(), self.status.reward, self.status.done, StepInfo(self.status, events)

    # ---------------------------------------------------------- observation
    def observe(self) -> dict:
        c, d = self.controller, self.data
        feats = self.tasks.features()
        obs = {
            "qpos": np.append(c.arm_qpos(d), c.finger_extension(d)),
            "qvel": np.append(c.arm_qvel(d), d.qvel[self.model.joint("finger").dofadr[0]]),
            "tool_pos": c.tool_pos(d),
            "tool_target": self.tool_target.copy(),
            "finger": c.finger_extension(d) / c.finger_max,
            "mouse_pos": self.devices.mouse_position(d),
            "cursor": np.array(self.desktop.cursor, dtype=float),
            "key_down": bool(self.devices.key_down.any()),
            "button_down": bool(self.devices.button_down),
            "palm_on_mouse": bool(self.devices.palm_on_mouse),
            "mouse_button": self.mouse_button_top(),
            **feats,
        }
        if self.config.render_obs:
            h, w = self.config.obs_image_size
            for cam in self.config.obs_cameras:
                obs[cam] = self.render(cam, h, w)
            obs["screen"] = self.screen_image(self.config.obs_screen_size)
        return obs

    def render(self, camera: str = "overview", height: int = 400, width: int = 640) -> np.ndarray:
        key = (height, width)
        if key not in self._renderers:
            self._renderers[key] = SceneRenderer(self.model, height, width, self.screen)
        return self._renderers[key].render(self.data, camera)

    def screen_image(self, size: tuple[int, int] | None = None) -> np.ndarray:
        """The simulated display as an (H, W, 3) array, optionally resized to size=(H, W)."""
        surf = self.desktop.render()
        if size is not None and (size[1], size[0]) != surf.get_size():
            surf = pygame.transform.smoothscale(surf, (size[1], size[0]))
        return pygame.surfarray.array3d(surf).transpose(1, 0, 2)

    def _sync_screen(self, force: bool = False) -> None:
        if force or self.desktop.version != self._screen_version:
            self.screen.write_surface(self.desktop.render())
            self._screen_version = self.desktop.version

    # ------------------------------------------------------------- metrics
    @property
    def elapsed(self) -> float:
        return self.data.time - self.tasks.start_time

    def metrics(self) -> dict:
        return {
            "task": self.task.name,
            "text": getattr(self.task, "text", None),
            "success": bool(self.status.success),
            "progress": float(self.status.progress),
            "time": self.tasks.success_time if self.status.success else self.elapsed,
            "steps": self.step_count,
            "wrong_keys": self.tasks.wrong_keys,
            "failed_clicks": self.tasks.failed_clicks,
            "collisions": self.devices.collision_count,
            "failed_contacts": self.devices.failed_contact_count,
            "key_presses": self.devices.press_count,
            "clicks": self.devices.click_count,
        }

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
