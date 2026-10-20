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

  click widget w
    a key or the button is down  -> retract and hold
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

The expert uses privileged simulator state (key positions, widget rectangles, cursor,
mouse position). `noise_scale` tells the data collector how much perturbation the current
phase tolerates: full while flying, little while pressing, almost none while dragging (the
cursor moves 6.4 px per millimetre of mouse travel).
"""

from __future__ import annotations

import numpy as np

from environment.computer_env import ComputerUseEnv

HOVER = 0.012  # palm height above the key tops while typing (m)
KEY_PRESS_CMD = 0.75  # finger command for key presses (~22 mm extension)
CLICK_CMD = 0.30  # finger command for mouse clicks (~9 mm extension)
FINGER_RETRACTED = 0.002  # m; finger counts as retracted below this extension
PALM_PRESS_DEPTH = 0.006  # target below the mouse surface so friction drags the mouse
GLIDE_HEIGHT = 0.015  # approach height above the press point far from the mouse
SOFT_LANDING = (0.02, 0.35)  # within 2 cm of the press point, cover 35% of the distance per step
LIFT_HEIGHT = 0.02  # how high to lift off the mouse before flying to the keyboard
PRESS_RAMP = (0.003, 0.006)  # full press within 3 mm of the key centre, none beyond 6 mm
DRAG_GAIN = 0.25  # fraction of the remaining cursor error corrected per control step
CLICK_RAMP_PX = (3.0, 6.0)  # full click within 3 px of the widget centre, none beyond 6 px
CURSOR_KEEP_PX = 8.0  # a click in progress finishes unless the cursor is pushed this far


def waypoint_action(offset: np.ndarray, finger: float, max_step: float) -> np.ndarray:
    """Action moving the tool target toward `offset` (waypoint - target) at up to max_step."""
    offset = np.asarray(offset, float)
    dist = np.linalg.norm(offset)
    if dist > max_step:
        offset = offset * (max_step / dist)
    return np.array([*offset, finger])


class ScriptedPolicy:
    name = "scripted"

    def __init__(self, env: ComputerUseEnv):
        self.env = env
        self.reset()

    def reset(self) -> None:
        self.skill = "idle"
        self.noise_scale = 1.0
        self.waypoint = self.env.tool_target.copy()
        self.finger = 0.0

    def act(self, obs: dict | None = None) -> np.ndarray:
        char, click = self.env.tasks.expectation()
        if click is not None:
            self.skill = f"click {click}"
            self._click(click)
        elif char is not None:
            self.skill = f"press {char!r}"
            self._type(char)
        else:
            self.skill = "idle"
            self._hold()
        return waypoint_action(self.waypoint - self.env.tool_target, self.finger, self.env.config.max_step)

    # ------------------------------------------------------------------ state
    def _hold(self) -> None:
        """Stay where the tool target is and retract the finger."""
        self.waypoint, self.finger, self.noise_scale = self.env.tool_target.copy(), 0.0, 0.3

    def _finger_out(self) -> bool:
        return self.env.controller.finger_extension(self.env.data) > FINGER_RETRACTED

    def _on_mouse(self, button: np.ndarray, surface: float) -> bool:
        """Palm touching the mouse. Mid-click, resting right above it also counts: pressing
        the button lifts the palm for a moment, which must not abort the click."""
        if self.env.devices.palm_on_mouse:
            return True
        tool = self.env.controller.tool_pos(self.env.data)
        near = np.linalg.norm(tool[:2] - button[:2]) < 0.005 and tool[2] < surface + 0.003
        return bool(near and self._finger_out())

    # ----------------------------------------------------------------- typing
    def _type(self, char: str) -> None:
        env, devices = self.env, self.env.devices
        if devices.palm_on_mouse:
            surface = env.mouse_rest_top()[2]
            self.waypoint = np.array([*env.tool_target[:2], surface + LIFT_HEIGHT])
            self.finger, self.noise_scale = 0.0, 0.3
            return
        if devices.key_down.any() or devices.button_down:
            return self._hold()
        hover = env.key_top(char) + [0, 0, HOVER]
        dist = np.linalg.norm(env.controller.tool_pos(env.data) - hover)
        near, far = PRESS_RAMP
        finger = KEY_PRESS_CMD * float(np.clip((far - dist) / (far - near), 0.0, 1.0))
        allowed = finger * env.controller.finger_max + FINGER_RETRACTED
        if env.controller.finger_extension(env.data) > allowed + 0.002:
            return self._hold()
        self.waypoint, self.finger = hover, finger
        self.noise_scale = 1.0 if dist > 0.01 else 0.3

    # --------------------------------------------------------------- clicking
    def _click(self, widget: str) -> None:
        env, devices = self.env, self.env.devices
        rect = env.desktop.widget_rect(widget)
        if rect is None or devices.key_down.any() or devices.button_down:
            return self._hold()

        button = env.mouse_button_top()
        surface = env.mouse_rest_top()[2]
        press_z = surface - PALM_PRESS_DEPTH
        if not self._on_mouse(button, surface):
            if self._finger_out():
                return self._hold()
            dist = np.linalg.norm(button[:2] - env.tool_target[:2])
            goal = np.array([button[0], button[1], press_z + min(GLIDE_HEIGHT, dist)])
            offset = goal - env.tool_target
            if np.linalg.norm(offset) < SOFT_LANDING[0]:
                offset *= SOFT_LANDING[1]
            self.waypoint = env.tool_target + offset
            self.finger, self.noise_scale = 0.0, (1.0 if dist > 0.03 else 0.1)
            return

        err = np.asarray(rect.center, float) - np.asarray(env.desktop.cursor, float)
        err_px = float(np.abs(err).max())
        if self._finger_out() and err_px < CURSOR_KEEP_PX:  # finish the click in progress
            self.waypoint = np.array([*env.tool_target[:2], press_z])
            self.finger, self.noise_scale = CLICK_CMD, 0.0
            return
        near, far = CLICK_RAMP_PX
        self.finger = CLICK_CMD * float(np.clip((far - err_px) / (far - near), 0.0, 1.0))
        self.noise_scale = 0.03
        if err_px > near:
            # Proportional servo on the cursor error. Inverse of the cursor mapping
            # (environment/ui_events.py): du = -g*dy, dv = -g*dx. Covering only part of the
            # error per step avoids overshoot: the hand lags its target under mouse friction.
            step = DRAG_GAIN * np.array([-err[1], -err[0]]) / env.bridge.gain
            self.waypoint = np.array([*(env.tool_target[:2] + step), press_z])
        else:
            self.waypoint = np.array([*env.tool_target[:2], press_z])
