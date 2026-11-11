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
    keyboard_pos (3,)            keyboard origin (its placement varies between scenes)
    instruction                  the task in words, e.g. 'Search the web for "mouse control"'
    expected_char, click_target, click_button, click_target_px
                                 task state (see TaskManager.features)

`annotations(camera)` returns ground-truth boxes and labels for every key, mouse button and
piece of screen text visible in a camera image (environment/annotations.py).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import pygame

from control.contact_detection import DeviceMonitor
from control.robot_controller import RobotController
from environment.browser import Desktop
from environment.dynamics import NOMINAL_DYNAMICS, HiddenDynamics
from environment.keyboard_layout import CAP_CENTER_Z, CAP_HALF_HEIGHT, KEY_BY_CHAR, KEYS
from environment.screen import SceneRenderer, ScreenTexture, paint_keycap_labels, paint_mouse_labels
from environment.task_manager import Task, TaskManager, TaskStatus, make_task
from environment.ui_events import PhysicalUIBridge, UIEvent
from environment.variation import NOMINAL, SceneVariation, VariationApplier

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
    # Tool-target limits; wide enough for every device placement in environment/variation.py.
    workspace_low: tuple[float, float, float] = (0.28, -0.55, 0.765)
    workspace_high: tuple[float, float, float] = (0.75, 0.36, 1.00)
    # "nominal": v1.0 physics. "realistic": the mouse can rotate on the pad and episodes can
    # carry hidden dynamics (environment/dynamics.py): latency, noise, sag, pointer
    # acceleration, per-key forces, auto-repeat, double-click timing.
    physics: str = "nominal"


# Realistic physics is slower (latency, sag, acceleration): every task gets 1.5x its time limit.
REALISTIC_TIME_SCALE = 1.5

# Rotation of a free mouse on its pad (realistic profile): friction and damping about z.
MOUSE_YAW_RANGE_DEG = 45.0
MOUSE_YAW_FRICTION = 0.02  # N m
MOUSE_YAW_DAMPING = 0.01  # N m s / rad
RIGID_CONTACT_SOLREF = (0.004, 1.0)  # realistic profile: stiff finger-key contacts (2x timestep)


def build_model(physics: str = "nominal") -> mujoco.MjModel:
    """The scene for a physics profile. Nominal is scene.xml exactly; the realistic profile
    adds the mouse's rotation joint."""
    if physics == "nominal":
        return mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    if physics != "realistic":
        raise ValueError(f"physics must be 'nominal' or 'realistic', not {physics!r}")
    spec = mujoco.MjSpec.from_file(str(SCENE_PATH))
    spec.body("mouse").add_joint(
        name="mouse_yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
        range=[-MOUSE_YAW_RANGE_DEG, MOUSE_YAW_RANGE_DEG],  # the scene compiles angles in degrees
        damping=MOUSE_YAW_DAMPING, frictionloss=MOUSE_YAW_FRICTION, armature=1e-4)
    model = spec.compile()
    # Rigid keycaps and buttons: the default soft contact on a 5 g key lets a firm press sink
    # the fingertip millimetres into the cap. Stiffen finger-key / finger-button contacts.
    hard = [model.geom(k.body).id for k in KEYS] + [model.geom(n).id for n in ("mouse_left", "mouse_right", "finger")]
    model.geom_solref[hard] = RIGID_CONTACT_SOLREF
    return model


@dataclass
class StepInfo:
    status: TaskStatus
    events: list[UIEvent] = field(default_factory=list)


class ComputerUseEnv:
    action_dim = 4

    def __init__(self, config: EnvConfig | None = None, task: Task | None = None):
        self.config = config or EnvConfig()
        self.model = build_model(self.config.physics)
        self.data = mujoco.MjData(self.model)
        self.realistic = self.config.physics == "realistic"
        self._yaw_qadr = self.model.joint("mouse_yaw").qposadr[0] if self.realistic else None
        self.n_substeps = max(1, round(self.config.control_dt / self.model.opt.timestep))

        self.controller = RobotController(self.model)
        self.devices = DeviceMonitor(self.model)
        self.desktop = Desktop()
        self.bridge = PhysicalUIBridge(self.devices, self.desktop, yaw_qadr=self._yaw_qadr)
        self.tasks = TaskManager(self.desktop, time_scale=REALISTIC_TIME_SCALE if self.realistic else 1.0)
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
        self._keyboard_body = self.model.body("keyboard").id
        self.tool_target = np.array(self.config.home_tool_pos, dtype=float)
        self.finger_cmd = 0.0
        self.variation = NOMINAL
        self._variations = VariationApplier(self)
        self.dynamics = NOMINAL_DYNAMICS
        self._arm_bodies = np.flatnonzero(self.model.body_gravcomp > 0)
        self._key_joints = self._variations.key_joints
        self._pending: deque = deque()
        self._frames: deque = deque(maxlen=1)
        self._rng = np.random.default_rng(0)

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

    def planned_tool_target(self) -> np.ndarray:
        """The hand's target once the commands already sent (but delayed by control latency)
        have taken effect; equal to `tool_target` without latency."""
        target = self.tool_target.copy()
        for a in self._pending:
            step = np.clip(a[:3], -self.config.max_step, self.config.max_step)
            target = np.clip(target + step, self._low, self._high)
        return target

    def mouse_yaw(self) -> float:
        """The mouse's rotation on the pad (always 0 with nominal physics)."""
        return float(self.data.qpos[self._yaw_qadr]) if self._yaw_qadr is not None else 0.0

    # ------------------------------------------------------------- core API
    def reset(self, task: Task | None = None, variation: SceneVariation | None = None,
              dynamics: HiddenDynamics | None = None) -> dict:
        """Start an episode. `variation` changes the scene (environment/variation.py) and
        `dynamics` the hidden physics (environment/dynamics.py, realistic profile only);
        either one, if omitted, stays as it was."""
        if task is not None:
            self.task = task
        if variation is not None and variation != self.variation:
            self._variations.apply(variation)
            self.variation = variation
            self._key_tops = self._compute_key_tops()
        if dynamics is not None:
            if dynamics != NOMINAL_DYNAMICS and not self.realistic:
                raise ValueError("hidden dynamics need EnvConfig(physics='realistic')")
            self.dynamics = dynamics
        self._apply_dynamics()
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if self._yaw_qadr is not None:
            d.qpos[self._yaw_qadr] = self.dynamics.mouse_yaw
        home = np.array(self.config.home_tool_pos, dtype=float)
        self.controller.reset(d, home)
        mujoco.mj_forward(m, d)
        self.tool_target = home
        self.finger_cmd = 0.0
        self.tasks.reset(self.task, d.time)
        self.bridge.reset(d)
        self._pending = deque()
        self._frames = deque(maxlen=self.dynamics.display_latency + 1)
        self._rng = np.random.default_rng(self.dynamics.seed)
        self._sync_screen(force=True)
        self.status = self.tasks.evaluate(d.time)
        self.step_count = 0
        return self.observe()

    def _apply_dynamics(self) -> None:
        """Write the hidden dynamics into the model, devices, desktop and mouse mapping."""
        dyn, m = self.dynamics, self.model
        m.body_gravcomp[self._arm_bodies] = dyn.gravcomp
        stiffness = self._variations.nominal["jnt_stiffness"][self._key_joints] * self.variation.key_stiffness_scale
        if dyn.key_stiffness is not None:
            stiffness = stiffness * np.asarray(dyn.key_stiffness)
        m.jnt_stiffness[self._key_joints] = stiffness
        self.devices.set_key_actuation(dyn.key_actuation)
        self.devices.autorepeat_delay = dyn.autorepeat_delay
        self.desktop.double_click_time = dyn.double_click_time
        self.bridge.pointer_accel = dyn.pointer_accel

    def step(self, action) -> tuple[dict, float, bool, StepInfo]:
        action = np.asarray(action, dtype=float)
        dyn = self.dynamics
        if dyn.action_delay:  # control latency: this step executes an earlier command
            self._pending.append(action.copy())
            action = (self._pending.popleft() if len(self._pending) > dyn.action_delay
                      else np.array([0.0, 0.0, 0.0, self.finger_cmd]))
        delta = np.clip(action[:3], -self.config.max_step, self.config.max_step)
        if dyn.action_noise:
            delta = delta + self._rng.normal(0.0, dyn.action_noise, 3)
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
        shown = self._frames[0] if self._frames else None  # what the (possibly lagging) monitor shows
        obs = {
            "instruction": self.task.instruction,
            "qpos": np.append(c.arm_qpos(d), c.finger_extension(d)),
            "qvel": np.append(c.arm_qvel(d), d.qvel[self.model.joint("finger").dofadr[0]]),
            "tool_pos": c.tool_pos(d),
            "tool_target": self.tool_target.copy(),
            "finger": c.finger_extension(d) / c.finger_max,
            "mouse_pos": self.devices.mouse_position(d),
            "cursor": np.array(shown["cursor"] if shown else self.desktop.cursor, dtype=float),
            "key_down": bool(self.devices.key_down.any()),
            "button_down": bool(self.devices.button_down),
            "palm_on_mouse": bool(self.devices.palm_on_mouse),
            "mouse_button": self.mouse_button_top(feats["click_button"] or "left"),
            "keyboard_pos": self.data.xpos[self._keyboard_body].copy(),
            **feats,
        }
        if self.dynamics.obs_noise:
            sigma = self.dynamics.obs_noise
            for key in ("tool_pos", "mouse_pos", "mouse_button", "keyboard_pos"):
                obs[key] = obs[key] + self._rng.normal(0.0, sigma, obs[key].shape)
            obs["qpos"] = obs["qpos"] + self._rng.normal(0.0, 5 * sigma, obs["qpos"].shape)
            obs["cursor"] = obs["cursor"] + self._rng.normal(0.0, 2000 * sigma, 2)  # ~1 px per 0.5 mm
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
        """The simulated display as an (H, W, 3) array (what the monitor shows, so it lags under
        display latency), optionally resized to size=(H, W)."""
        surf = self._frames[0]["surface"] if self._frames else self.desktop.render()
        if size is not None and (size[1], size[0]) != surf.get_size():
            surf = pygame.transform.smoothscale(surf, (size[1], size[0]))
        return pygame.surfarray.array3d(surf).transpose(1, 0, 2)

    def _sync_screen(self, force: bool = False) -> None:
        latency = self.dynamics.display_latency
        if latency == 0:
            if force or self.desktop.version != self._screen_version:
                self.screen.write_surface(self.desktop.render())
                self._screen_version = self.desktop.version
            return
        # Display latency: the monitor shows the frame from `latency` control steps ago.
        if force or not self._frames or self._frames[-1]["version"] != self.desktop.version:
            frame = {"version": self.desktop.version, "surface": self.desktop.render().copy(),
                     "cursor": self.desktop.cursor}
        else:
            frame = self._frames[-1]
        if force:
            self._frames.clear()
        self._frames.append(frame)
        shown = self._frames[0]
        if force or shown["version"] != self._screen_version:
            self.screen.write_surface(shown["surface"])
            self._screen_version = shown["version"]

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
