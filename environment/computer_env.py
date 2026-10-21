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
    robot_cam / keyboard_cam / ... RGB images from `obs_cameras` (if render_obs); sizes per
                                 camera from `obs_image_size` (default: CAMERA_SIZES)
    screen                       RGB image of the simulated display (if render_obs)
    qpos, qvel                   arm joints + finger (6,)
    tool_pos, tool_target        (3,)
    mouse_pos (2,) metres, cursor (2,) pixels, key_down, button_down, palm_on_mouse
    mouse_button (3,)            top of the mouse button the task needs next (left by default)
    expected_char, click_target, click_button, click_target_px
                                 task state (see TaskManager.features)

`annotations(camera)` returns ground-truth boxes and labels for every key, mouse button and
piece of screen text visible in a camera image (environment/annotations.py).
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
from environment.screen import SceneRenderer, ScreenTexture, paint_keycap_labels, paint_mouse_labels
from environment.task_manager import Task, TaskManager, TaskStatus, make_task
from environment.ui_events import PhysicalUIBridge, UIEvent

SCENE_PATH = Path(__file__).resolve().parent / "scene.xml"

# Default (H, W) per camera. The device cameras are sized so legends and screen text stay
# legible: ~32 px per key in keyboard_cam, ~35 px per mouse button in mouse_cam, and
# screen_cam 1:1 with the 640x400 screen texture.
CAMERA_SIZES: dict[str, tuple[int, int]] = {
    "robot_cam": (240, 320),
    "wrist_cam": (240, 320),
    "keyboard_cam": (256, 512),
    "mouse_cam": (320, 320),
    "screen_cam": (400, 640),
    "overview": (400, 640),
    "front_view": (400, 640),
    "side_view": (400, 640),
    "top_view": (400, 640),
}


@dataclass
class EnvConfig:
    control_dt: float = 0.02  # 50 Hz control
    max_step: float = 0.006  # m per control step (0.3 m/s)
    obs_cameras: tuple[str, ...] = ("robot_cam",)
    # (H, W) for every camera, a {camera: (H, W)} dict, or None for CAMERA_SIZES.
    obs_image_size: tuple[int, int] | dict[str, tuple[int, int]] | None = None
    obs_screen_size: tuple[int, int] = (400, 640)  # (H, W); native resolution keeps text legible
    render_obs: bool = True  # False -> no images in observations (fast data collection)
    # Home pose: beside the keyboard, so the top-down keyboard camera sees every key at reset.
    home_tool_pos: tuple[float, float, float] = (0.40, -0.12, 0.86)
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
        paint_mouse_labels(self.model)
        self._renderers: dict[tuple[int, int], SceneRenderer] = {}
        self._screen_version = -1

        self.task = task or make_task("type_key", "a")
        self._low = np.array(self.config.workspace_low)
        self._high = np.array(self.config.workspace_high)
        self._button_sites = {b: self.model.site(f"mouse_{b}_top").id for b in ("left", "right")}
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

    def mouse_button_top(self, button: str = "left") -> np.ndarray:
        """World position of the centre of a mouse button's top face."""
        return self.data.site_xpos[self._button_sites[button]].copy()

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
            "mouse_button": self.mouse_button_top(feats["click_button"] or "left"),
            **feats,
        }
        if self.config.render_obs:
            for cam in self.config.obs_cameras:
                obs[cam] = self.render(cam, *self.camera_size(cam))
            obs["screen"] = self.screen_image(self.config.obs_screen_size)
        return obs

    def camera_size(self, camera: str) -> tuple[int, int]:
        """(H, W) used for `camera` in observations."""
        size = self.config.obs_image_size
        if isinstance(size, dict):
            size = size.get(camera)
        return tuple(size or CAMERA_SIZES.get(camera, (240, 320)))

    def _renderer(self, height: int, width: int) -> SceneRenderer:
        key = (height, width)
        if key not in self._renderers:
            self._renderers[key] = SceneRenderer(self.model, height, width, self.screen)
        return self._renderers[key]

    def render(self, camera="overview", height: int = 400, width: int = 640) -> np.ndarray:
        return self._renderer(height, width).render(self.data, camera)

    def segmentation(self, camera, height: int = 400, width: int = 640) -> np.ndarray:
        """(H, W) geom id per pixel (-1 for background) from `camera`."""
        return self._renderer(height, width).segmentation(self.data, camera)

    def annotations(self, camera: str, height: int | None = None, width: int | None = None) -> list[dict]:
        """Ground-truth boxes/labels of keys, mouse buttons, screen text, ... seen by `camera`."""
        from environment.annotations import annotate

        if height is None or width is None:
            height, width = self.camera_size(camera)
        return annotate(self, camera, height, width)

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
