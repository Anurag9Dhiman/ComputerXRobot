"""A non-privileged expert: the oracle's control law on dynamics identified online.

The oracle (learning/scripted_policy.py with SimWorld) reads hidden quantities from the
simulator. `ObservedWorld` replaces each one with something a `state`-track agent can know:
its observations, its own past commands, and fixed public facts (the keyboard layout, the
mouse's shape, the controller's limits). The hidden dynamics (environment/dynamics.py) are
identified within the episode, from how the world responds:

    control latency     The observed hand target (`tool_target`) moves by the command sent
                        d steps earlier. d is the lag under which replaying the agent's own
                        commands best reproduces the target's motion; the last d commands are
                        still in flight, which gives the planned target the law works from.
    display latency,    The cursor seen on screen lags the mouse by L frames and moves by
    cursor mapping      J @ (mouse displacement), J a 2x2 matrix (gain x mouse rotation). J is
                        fitted by recursive least squares for every candidate L, and the L
                        with the smallest prediction error is used. The current cursor is
                        predicted from the lagged one plus J times the mouse motion since
                        (a Smith predictor).
    pointer accel.,     J is refitted with forgetting, so it follows the gain as the drag
    mouse rotation      speed and the mouse's rotation change.
    observation noise   Measured from the keyboard's observed position, which never moves.
                        Static quantities (keyboard position, mouse height) are averaged.
    arm sag             As the oracle: the tracking error between target and hand.

Per-key force and actuation depth, auto-repeat and double-click time need no estimate: the
law does not use them (it presses firmly, retracts as soon as a key is down, and finishes a
click in progress at once, so a second click follows within the double-click time).

    python -m benchmark run --agent adaptive --split hard_test_ood --workers 8
    python -m benchmark run --agent fixed --split hard_test_ood --workers 8   # no identification
"""

from __future__ import annotations

from collections import deque

import mujoco
import numpy as np

from control.robot_controller import RobotController
from environment.browser import SCREEN_H, SCREEN_W
from environment.computer_env import EnvConfig, build_model
from environment.keyboard_layout import CAP_CENTER_Z, CAP_HALF_HEIGHT, KEY_BY_CHAR
from environment.variation import NOMINAL_GAIN
from learning.scripted_policy import DRAG_GAIN, ScriptedPolicy

DELAYS = 7  # control-latency hypotheses: 0..6 steps (the hard tier draws up to 4)
LAGS = 8  # display-latency hypotheses: 0..7 frames (the hard tier draws up to 5)
WINDOW = 4  # control steps per cursor-mapping sample
MIN_MOTION = 0.003  # m of mouse travel for a cursor-mapping sample
FORGET = 0.97  # RLS forgetting factor of the cursor mapping (pointer acceleration, rotation)
PRIOR = 1e5  # RLS prior covariance of J (1/m^2): the first ~4 mm sample weighs as much as J0
IDENTIFIED = 6  # cursor-mapping samples before the display latency counts as identified
NOISE_PRIOR = 0.001  # m; assumed observation noise until the keyboard has been seen a few times
SMITH_THRESHOLD = 3.0  # predicted cursor motion below this many noise std devs is ignored
J0 = np.array([[0.0, -NOMINAL_GAIN], [-NOMINAL_GAIN, 0.0]])  # nominal mouse: gain 6400 px/m, no yaw


class _CursorModel:
    """cursor displacement ~= J @ mouse displacement, by recursive least squares."""

    def __init__(self):
        self.J = J0.copy()
        self.P = np.eye(2) * PRIOR
        self.sq_err = 0.0
        self.n = 0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        e = y - self.J @ x  # a-priori error: scores this lag hypothesis
        self.sq_err += float(e @ e)
        self.n += 1
        px = self.P @ x
        k = px / (FORGET + x @ px)
        self.J = self.J + np.outer(e, k)
        self.P = (self.P - np.outer(k, px)) / FORGET
        if np.trace(self.P) > 2 * PRIOR:  # no wind-up while the mouse rests
            self.P *= 2 * PRIOR / np.trace(self.P)

    @property
    def mean_err(self) -> float:
        return self.sq_err / self.n if self.n else np.inf


class ObservedWorld:
    """The view of ScriptedPolicy for a non-privileged agent (see the module docstring).

    Call `observe(obs)` before and `record(action)` after every `ScriptedPolicy.act()`.
    With identify=False it assumes the nominal dynamics (no latency, the nominal cursor
    mapping) instead of estimating them: a control for what identification contributes."""

    def __init__(self, config: EnvConfig, identify: bool = True):
        self.identify = identify
        self.realistic = config.physics == "realistic"
        self.max_step = config.max_step
        self._low, self._high = np.array(config.workspace_low), np.array(config.workspace_high)
        self._home = np.array(config.home_tool_pos, float)
        # Public geometry, from a freshly built canonical scene (not from the episode's env).
        model = build_model(config.physics)
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        self.finger_max = RobotController(model).finger_max
        keyboard = data.xpos[model.body("keyboard").id]
        top = np.array([0.0, 0.0, CAP_CENTER_Z + CAP_HALF_HEIGHT])
        self._key_offsets = {c: data.xpos[model.body(k.body).id] + top - keyboard for c, k in KEY_BY_CHAR.items()}
        rest_z = data.site_xpos[model.site("mouse_rest_top").id][2]
        self._button_above_rest = {b: data.site_xpos[model.site(f"mouse_{b}_top").id][2] - rest_z
                                   for b in ("left", "right")}
        self.reset()

    # ------------------------------------------------------------- episode
    def reset(self) -> None:
        self.obs: dict | None = None
        self._targets: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._delay_err = np.zeros(DELAYS)
        self._cursor_models = [_CursorModel() for _ in range(LAGS)]
        self._mouse = deque(maxlen=LAGS + WINDOW + 1)
        self._cursor = deque(maxlen=LAGS + WINDOW + 1)
        self._kb_n, self._kb_mean, self._kb_m2 = 0, np.zeros(3), np.zeros(3)
        self._surface_n, self._surface = 0, 0.0

    def observe(self, obs: dict) -> None:
        self.obs = obs
        target = np.asarray(obs["tool_target"], float)
        if self._targets:
            self._update_delay(self._targets[-1], target)
        self._targets.append(target)
        self._update_static(obs)
        self._mouse.append(np.asarray(obs["mouse_pos"], float))
        self._cursor.append(np.asarray(obs["cursor"], float))
        self._update_cursor_models()

    def record(self, action: np.ndarray) -> None:
        self._actions.append(np.asarray(action, float).copy())

    # ---------------------------------------------------------- estimators
    def _step(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action[:3], -self.max_step, self.max_step)

    def _update_delay(self, before: np.ndarray, after: np.ndarray) -> None:
        """Score every control-latency hypothesis d: the target moved by the command sent d
        steps before the last one (nothing, if there was none yet)."""
        t = len(self._actions)  # the step just executed was the t-th command sent
        for d in range(DELAYS):
            i = t - 1 - d
            step = self._step(self._actions[i]) if i >= 0 else np.zeros(3)
            self._delay_err[d] += float(np.sum((after - np.clip(before + step, self._low, self._high)) ** 2))

    @property
    def control_delay(self) -> int:
        if not self.identify:
            return 0
        return int(np.argmin(self._delay_err))  # ties (no evidence yet) -> the shortest

    def _update_static(self, obs: dict) -> None:
        kb = np.asarray(obs["keyboard_pos"], float)
        self._kb_n += 1
        delta = kb - self._kb_mean
        self._kb_mean += delta / self._kb_n
        self._kb_m2 += delta * (kb - self._kb_mean)
        if not obs["button_down"]:
            z = float(obs["mouse_button"][2]) - self._button_above_rest[obs.get("click_button") or "left"]
            self._surface_n += 1
            self._surface += (z - self._surface) / self._surface_n

    @property
    def obs_noise(self) -> float:
        """Std of the position observations, from the keyboard (which never moves)."""
        if self._kb_n < 5:
            return NOISE_PRIOR
        return float(np.sqrt(np.mean(self._kb_m2[:2]) / (self._kb_n - 1)))

    def _update_cursor_models(self) -> None:
        """One cursor-mapping sample per lag hypothesis L: the cursor's motion over the last
        WINDOW frames against the mouse's motion over the WINDOW steps ending L steps ago."""
        if len(self._cursor) < LAGS + WINDOW + 1:
            return
        cursor, mouse = list(self._cursor), list(self._mouse)
        window = np.array(cursor[-WINDOW - 1:])
        if (window[:, 0] < 1).any() or (window[:, 0] > SCREEN_W - 2).any() or \
                (window[:, 1] < 1).any() or (window[:, 1] > SCREEN_H - 2).any():
            return  # the cursor is pinned at a screen edge: the mapping does not hold
        y = cursor[-1] - cursor[-1 - WINDOW]
        xs = [mouse[-1 - L] - mouse[-1 - L - WINDOW] for L in range(LAGS)]
        if max(np.linalg.norm(x) for x in xs) < MIN_MOTION:
            return
        for model, x in zip(self._cursor_models, xs):
            model.update(x, y)

    @property
    def display_lag(self) -> int:
        if not self.identify:
            return 0
        return int(np.argmin([m.mean_err for m in self._cursor_models]))

    @property
    def lag_identified(self) -> bool:
        return not self.identify or self._cursor_models[0].n >= IDENTIFIED

    def _mapping(self) -> np.ndarray:
        if not self.identify:
            return J0
        J = self._cursor_models[self.display_lag].J if self.lag_identified else J0
        gain = np.sqrt(abs(np.linalg.det(J)))
        if not 0.3 * NOMINAL_GAIN < gain < 5 * NOMINAL_GAIN:
            return J0
        return J

    # ------------------------------------------------------- view interface
    def expectation(self) -> tuple[str | None, str | None]:
        return self.obs["expected_char"], self.obs["click_target"]

    def planned_target(self) -> np.ndarray:
        if self.obs is None:
            return self._home.copy()
        target = np.asarray(self.obs["tool_target"], float).copy()
        d = self.control_delay
        for action in self._actions[len(self._actions) - d:] if d else []:
            target = np.clip(target + self._step(action), self._low, self._high)
        return target

    def tool_target(self) -> np.ndarray:
        return np.asarray(self.obs["tool_target"], float)

    def tool_pos(self) -> np.ndarray:
        return np.asarray(self.obs["tool_pos"], float)

    def finger_extension(self) -> float:
        return float(self.obs["finger"]) * self.finger_max

    def palm_on_mouse(self) -> bool:
        return bool(self.obs["palm_on_mouse"])

    def device_down(self) -> bool:
        return bool(self.obs["key_down"] or self.obs["button_down"])

    def key_top(self, char: str) -> np.ndarray:
        return self._kb_mean + self._key_offsets[char]

    def mouse_button_top(self, button: str) -> np.ndarray:
        return np.asarray(self.obs["mouse_button"], float)

    def mouse_surface(self) -> float:
        return self._surface

    def widget_center(self, widget: str | None) -> np.ndarray | None:
        center = self.obs["click_target_px"]
        return None if widget is None or center is None else np.asarray(center, float)

    def cursor(self) -> np.ndarray:
        """The cursor now: the displayed (lagging) cursor plus the motion since, predicted
        from the mouse's motion over the display latency (a Smith predictor)."""
        cursor = np.asarray(self.obs["cursor"], float)
        L = self.display_lag if self.lag_identified else 0
        if L == 0 or len(self._mouse) <= L:
            return cursor
        J = self._mapping()
        motion = J @ (self._mouse[-1] - self._mouse[-1 - L])
        # Mouse observations are noisy: shrink the correction by its noise level, so a resting
        # mouse does not make the predicted cursor jitter.
        noise = SMITH_THRESHOLD * np.sqrt(2) * self.obs_noise * np.linalg.norm(J, 2)
        size = np.linalg.norm(motion)
        if size <= noise:
            return cursor
        return np.clip(cursor + motion * (1 - noise / size), 0, [SCREEN_W - 1, SCREEN_H - 1])

    def drag_step(self, err: np.ndarray) -> np.ndarray:
        """As the oracle's, with the estimated mapping J and latency: k * J^-1 @ err. Until the
        display latency is identified the loop gain allows for the longest one."""
        lag = 0 if self.lag_identified else LAGS - 1
        k = DRAG_GAIN / (1 + self.control_delay + lag)
        return k * np.linalg.solve(self._mapping(), err)


class AdaptiveExpert:
    """ScriptedPolicy on an ObservedWorld: act(obs) -> action, from the `state` track."""

    def __init__(self, config: EnvConfig, identify: bool = True):
        self.world = ObservedWorld(config, identify)
        self.policy = ScriptedPolicy(world=self.world)

    def reset(self) -> None:
        self.world.reset()
        self.policy.reset()

    def act(self, obs: dict) -> np.ndarray:
        self.world.observe(obs)
        action = self.policy.act()
        self.world.record(action)
        return action
