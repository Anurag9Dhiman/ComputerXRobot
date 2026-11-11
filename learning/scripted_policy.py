"""Scripted expert for every task.

The expert is reactive and memoryless: every control step it looks at the task's
expectation (character to type next or widget to click next) and the device state, and
picks a waypoint for the tool plus a finger command:

  type character c
    palm on the mouse            -> lift off the mouse
    a key is down                -> retract the finger and hold still
    finger out further than the  -> retract and hold (never sweep an extended finger
      distance allows               across the keyboard)
    otherwise                    -> fly straight to the hover point above c's key; the
                                    finger command ramps from 0 (6 mm away) to a full
                                    press (3 mm away). Partial extension cannot reach the
                                    keycaps from hover height, and the smooth ramp is much
                                    easier for a learned policy to imitate than a step.

  click widget w with the left or right button (target "w" or "right:w")
    a key or a button is down    -> retract and hold
    palm on the mouse, finger    -> lift off and re-land with the finger over the other
      over the other button         button
    hand not on the mouse        -> glide toward the mouse; the target height drops as the
                                    hand gets close and the last 2 cm are approached
                                    geometrically, so the palm lands softly on the mouse
    cursor not on w              -> drag: servo the hand on (cursor error / mouse gain),
                                    pressing down so friction carries the mouse along
    cursor on w                  -> press the button with the finger; as with keys, the
                                    finger command ramps up as the cursor error shrinks

Because the waypoint and finger command are functions of the current state (no hidden
phase), the same rule recovers from slip, perturbations and mistakes -- a typo makes the
task expect a backspace, a missed click is simply retried -- and imitation learning can fit
it directly (learning/dataset.py stores the waypoint as the training label).

The expert reads the world through a *view*. `SimWorld` is the oracle's: privileged
simulator state (key positions, widget rectangles, the true cursor, the commands still in
flight, the true cursor gain and mouse rotation). `learning/adaptive.py` provides a view
built from state-track observations and online estimates of the hidden dynamics instead,
which turns the same control law into a non-privileged agent. `noise_scale` tells the data
collector how much perturbation the current phase tolerates: full while flying, 30% while
pressing keys and dragging the mouse (so demonstrations include recovering from cursor
overshoot), none during a click.
"""

from __future__ import annotations

import numpy as np

from environment.browser import split_click
from environment.computer_env import ComputerUseEnv

HOVER = 0.012  # palm height above the key tops while typing (m)
KEY_PRESS_CMD = 0.75  # finger command for key presses (~22 mm extension)
REALISTIC_KEY_PRESS_CMD = 1.0  # firmer, for the stiffer / deeper keys of the realistic profile
CLICK_CMD = 0.50  # finger command for mouse clicks (~15 mm; buttons sit 3 mm below the palm rest)
FINGER_RETRACTED = 0.002  # m; finger counts as retracted below this extension
PALM_PRESS_DEPTH = 0.006  # target below the mouse surface so friction drags the mouse
GLIDE_HEIGHT = 0.015  # approach height above the press point far from the mouse
SOFT_LANDING = (0.02, 0.35, 0.002)  # within 2 cm of the press point: 35% of the distance per
# step, but at least 2 mm, so the motion never fades to nothing just above the mouse
LIFT_HEIGHT = 0.02  # how high to lift off the mouse before flying to the keyboard
PRESS_RAMP = (0.003, 0.006)  # full press within 3 mm of the key centre, none beyond 6 mm
DRAG_GAIN = 0.25  # fraction of the remaining cursor error corrected per control step
CLICK_RAMP_PX = (3.0, 6.0)  # full click within 3 px of the widget centre, none beyond 6 px
CURSOR_KEEP_PX = 8.0  # a click in progress finishes unless the cursor is pushed this far
WRONG_BUTTON_DIST = 0.01  # m; the buttons' centres are 36.5 mm apart
SAG_COMPENSATION_RADIUS = 0.02  # m; realistic physics: correct for arm sag near a key
REALISTIC_DRAG_MAX = 0.002  # m per step; realistic physics: slower drags, less pointer acceleration


def waypoint_action(offset: np.ndarray, finger: float, max_step: float) -> np.ndarray:
    """Action moving the tool target toward `offset` (waypoint - target) at up to max_step."""
    offset = np.asarray(offset, float)
    dist = np.linalg.norm(offset)
    if dist > max_step:
        offset = offset * (max_step / dist)
    return np.array([*offset, finger])


class SimWorld:
    """The oracle's view: reads the simulator directly (privileged).

    Every quantity the expert's control law needs comes through a view with these members,
    so a non-privileged agent can run the same law on estimates (learning/adaptive.py)."""

    def __init__(self, env: ComputerUseEnv):
        self.env = env

    @property
    def realistic(self) -> bool:
        return self.env.realistic

    @property
    def max_step(self) -> float:
        return self.env.config.max_step

    @property
    def finger_max(self) -> float:
        return self.env.controller.finger_max

    def expectation(self) -> tuple[str | None, str | None]:
        return self.env.tasks.expectation()

    def planned_target(self) -> np.ndarray:
        """Where the hand's target will be once the commands in flight (control latency)
        have taken effect."""
        return self.env.planned_tool_target()

    def tool_target(self) -> np.ndarray:
        return self.env.tool_target

    def tool_pos(self) -> np.ndarray:
        return self.env.controller.tool_pos(self.env.data)

    def finger_extension(self) -> float:
        return self.env.controller.finger_extension(self.env.data)

    def palm_on_mouse(self) -> bool:
        return self.env.devices.palm_on_mouse

    def device_down(self) -> bool:
        """A key or a mouse button is held down."""
        return bool(self.env.devices.key_down.any() or self.env.devices.button_down)

    def key_top(self, char: str) -> np.ndarray:
        return self.env.key_top(char)

    def mouse_button_top(self, button: str) -> np.ndarray:
        return self.env.mouse_button_top(button)

    def mouse_surface(self) -> float:
        """Height of the mouse's palm rest."""
        return self.env.mouse_rest_top()[2]

    def widget_center(self, widget: str | None) -> np.ndarray | None:
        rect = self.env.desktop.widget_rect(widget) if widget else None
        return None if rect is None else np.asarray(rect.center, float)

    def cursor(self) -> np.ndarray:
        return np.asarray(self.env.desktop.cursor, float)

    def drag_step(self, err: np.ndarray) -> np.ndarray:
        """Hand displacement (x, y) that moves the cursor by a fraction of `err` (pixels).

        Inverse of the cursor mapping (environment/ui_events.py): du = -g*dy_m, dv = -g*dx_m
        in the mouse's own frame, rotated into the world by the mouse's yaw. Covering only
        part of the error per step avoids overshoot: the hand lags its target under mouse
        friction."""
        env = self.env
        if env.realistic:
            # Privileged: the true cursor gain at the current mouse speed (pointer
            # acceleration) and a loop gain reduced by the control latency to stay stable.
            gain = env.bridge.effective_gain()
            k = DRAG_GAIN / (1 + env.dynamics.action_delay)
        else:
            gain, k = env.bridge.gain, DRAG_GAIN
        bx, by = k * np.array([-err[1], -err[0]]) / gain
        c, s = np.cos(env.mouse_yaw()), np.sin(env.mouse_yaw())
        return np.array([c * bx - s * by, s * bx + c * by])


class ScriptedPolicy:
    name = "scripted"

    def __init__(self, env: ComputerUseEnv | None = None, world=None):
        """The oracle on `env` (SimWorld), or the same control law on another view."""
        if world is None and env is None:
            raise ValueError("ScriptedPolicy needs an env or a world view")
        self.env = env
        self.world = world if world is not None else SimWorld(env)
        self.reset()

    def reset(self) -> None:
        self.skill = "idle"
        self.noise_scale = 1.0
        self.waypoint = self._target.copy()
        self.finger = 0.0

    def act(self, obs: dict | None = None) -> np.ndarray:
        char, click = self.world.expectation()
        if click is not None:
            widget, button = split_click(click)
            self.skill = f"{'right-click' if button == 'right' else 'click'} {widget}"
            self._click(click)
        elif char is not None:
            self.skill = f"press {char!r}"
            self._type(char)
        else:
            self.skill = "idle"
            self._hold()
        return waypoint_action(self.waypoint - self._target, self.finger, self.world.max_step)

    # ------------------------------------------------------------------ state
    @property
    def _target(self) -> np.ndarray:
        """Where the hand's target will be after the commands already in flight (control
        latency); the expert plans from there so it never re-issues a pending correction."""
        return self.world.planned_target()

    def _hold(self) -> None:
        """Stay where the tool target is and retract the finger."""
        self.waypoint, self.finger, self.noise_scale = self._target.copy(), 0.0, 0.3

    def _finger_out(self) -> bool:
        return self.world.finger_extension() > FINGER_RETRACTED

    def _on_mouse(self, button: np.ndarray, surface: float) -> bool:
        """Palm touching the mouse. Mid-click, resting right above it also counts: pressing
        the button lifts the palm for a moment, which must not abort the click."""
        if self.world.palm_on_mouse():
            return True
        tool = self.world.tool_pos()
        near = np.linalg.norm(tool[:2] - button[:2]) < 0.005 and tool[2] < surface + 0.003
        return bool(near and self._finger_out())

    # ----------------------------------------------------------------- typing
    def _type(self, char: str) -> None:
        w = self.world
        if w.palm_on_mouse():
            self.waypoint = np.array([*self._target[:2], w.mouse_surface() + LIFT_HEIGHT])
            self.finger, self.noise_scale = 0.0, 0.3
            return
        if w.device_down():
            return self._hold()
        hover = w.key_top(char) + [0, 0, HOVER]
        tool = w.tool_pos()
        dist = np.linalg.norm(tool - hover)
        near, far = PRESS_RAMP
        press = REALISTIC_KEY_PRESS_CMD if w.realistic else KEY_PRESS_CMD
        finger = press * float(np.clip((far - dist) / (far - near), 0.0, 1.0))
        allowed = finger * w.finger_max + FINGER_RETRACTED
        if w.finger_extension() > allowed + 0.002:
            return self._hold()
        goal = hover
        if w.realistic and dist < SAG_COMPENSATION_RADIUS:
            # Under imperfect gravity compensation the hand settles below its target; aim
            # higher by the current tracking error so it settles on the hover point.
            goal = hover + (w.tool_target() - tool)
        self.waypoint, self.finger = goal, finger
        self.noise_scale = 1.0 if dist > 0.01 else 0.3

    # --------------------------------------------------------------- clicking
    def _click(self, target: str) -> None:
        w = self.world
        widget, which = split_click(target)
        center = w.widget_center(widget)
        if center is None or w.device_down():
            return self._hold()

        button = w.mouse_button_top(which)
        surface = w.mouse_surface()
        press_z = surface - PALM_PRESS_DEPTH
        hand = w.tool_pos()
        if w.palm_on_mouse() and not self._finger_out() and np.linalg.norm(hand[:2] - button[:2]) > WRONG_BUTTON_DIST:
            self.waypoint = np.array([*self._target[:2], surface + LIFT_HEIGHT])  # wrong button: lift off
            self.finger, self.noise_scale = 0.0, 0.3
            return
        if not self._on_mouse(button, surface):
            if self._finger_out():
                return self._hold()
            dist = np.linalg.norm(button[:2] - self._target[:2])
            goal = np.array([button[0], button[1], press_z + min(GLIDE_HEIGHT, dist)])
            offset = goal - self._target
            dist_3d = np.linalg.norm(offset)
            if dist_3d < SOFT_LANDING[0]:
                offset *= max(SOFT_LANDING[1], min(1.0, SOFT_LANDING[2] / max(dist_3d, 1e-9)))
            self.waypoint = self._target + offset
            self.finger, self.noise_scale = 0.0, (1.0 if dist > 0.03 else 0.1)
            return

        err = center - w.cursor()
        err_px = float(np.abs(err).max())
        if self._finger_out() and err_px < CURSOR_KEEP_PX:  # finish the click in progress
            self.waypoint = np.array([*self._target[:2], press_z])
            self.finger, self.noise_scale = CLICK_CMD, 0.0
            return
        near, far = CLICK_RAMP_PX
        self.finger = CLICK_CMD * float(np.clip((far - err_px) / (far - near), 0.0, 1.0))
        self.noise_scale = 0.3
        if err_px > near:
            # Proportional servo on the cursor error; with realistic physics the drag speed is
            # capped so pointer acceleration stays moderate.
            step = w.drag_step(err)
            if w.realistic and np.linalg.norm(step) > REALISTIC_DRAG_MAX:
                step *= REALISTIC_DRAG_MAX / np.linalg.norm(step)
            self.waypoint = np.array([*(self._target[:2] + step), press_z])
        else:
            self.waypoint = np.array([*self._target[:2], press_z])
