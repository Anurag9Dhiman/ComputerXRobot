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

The expert uses privileged simulator state (key positions, widget rectangles, cursor,
mouse position). `noise_scale` tells the data collector how much perturbation the current
phase tolerates: full while flying, 30% while pressing keys and dragging the mouse (so
demonstrations include recovering from cursor overshoot), none during a click.
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


class ScriptedPolicy:
    name = "scripted"

    def __init__(self, env: ComputerUseEnv):
        self.env = env
        self.reset()

    def reset(self) -> None:
        self.skill = "idle"
        self.noise_scale = 1.0
        self.waypoint = self._target.copy()
        self.finger = 0.0

    def act(self, obs: dict | None = None) -> np.ndarray:
        char, click = self.env.tasks.expectation()
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
        return waypoint_action(self.waypoint - self._target, self.finger, self.env.config.max_step)

    # ------------------------------------------------------------------ state
    @property
    def _target(self) -> np.ndarray:
        """Where the hand's target will be after the commands already in flight (control
        latency); the expert plans from there so it never re-issues a pending correction."""
        return self.env.planned_tool_target()

    def _hold(self) -> None:
        """Stay where the tool target is and retract the finger."""
        self.waypoint, self.finger, self.noise_scale = self._target.copy(), 0.0, 0.3

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
            self.waypoint = np.array([*self._target[:2], surface + LIFT_HEIGHT])
            self.finger, self.noise_scale = 0.0, 0.3
            return
        if devices.key_down.any() or devices.button_down:
            return self._hold()
        hover = env.key_top(char) + [0, 0, HOVER]
        dist = np.linalg.norm(env.controller.tool_pos(env.data) - hover)
        near, far = PRESS_RAMP
        press = REALISTIC_KEY_PRESS_CMD if env.realistic else KEY_PRESS_CMD
        finger = press * float(np.clip((far - dist) / (far - near), 0.0, 1.0))
        allowed = finger * env.controller.finger_max + FINGER_RETRACTED
        if env.controller.finger_extension(env.data) > allowed + 0.002:
            return self._hold()
        goal = hover
        if env.realistic and dist < SAG_COMPENSATION_RADIUS:
            # Under imperfect gravity compensation the hand settles below its target; aim
            # higher by the current tracking error so it settles on the hover point.
            goal = hover + (env.tool_target - env.controller.tool_pos(env.data))
        self.waypoint, self.finger = goal, finger
        self.noise_scale = 1.0 if dist > 0.01 else 0.3

    # --------------------------------------------------------------- clicking
    def _click(self, target: str) -> None:
        env, devices = self.env, self.env.devices
        widget, which = split_click(target)
        rect = env.desktop.widget_rect(widget) if widget else None
        if rect is None or devices.key_down.any() or devices.button_down:
            return self._hold()

        button = env.mouse_button_top(which)
        surface = env.mouse_rest_top()[2]
        press_z = surface - PALM_PRESS_DEPTH
        hand = env.controller.tool_pos(env.data)
        if devices.palm_on_mouse and not self._finger_out() and np.linalg.norm(hand[:2] - button[:2]) > WRONG_BUTTON_DIST:
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

        err = np.asarray(rect.center, float) - np.asarray(env.desktop.cursor, float)
        err_px = float(np.abs(err).max())
        if self._finger_out() and err_px < CURSOR_KEEP_PX:  # finish the click in progress
            self.waypoint = np.array([*self._target[:2], press_z])
            self.finger, self.noise_scale = CLICK_CMD, 0.0
            return
        near, far = CLICK_RAMP_PX
        self.finger = CLICK_CMD * float(np.clip((far - err_px) / (far - near), 0.0, 1.0))
        self.noise_scale = 0.3
        if err_px > near:
            # Proportional servo on the cursor error. Inverse of the cursor mapping
            # (environment/ui_events.py): du = -g*dy_m, dv = -g*dx_m in the mouse's own frame,
            # rotated into the world by the mouse's yaw. Covering only part of the error per
            # step avoids overshoot: the hand lags its target under mouse friction.
            if env.realistic:
                # Privileged: the true cursor gain at the current mouse speed (pointer
                # acceleration), a loop gain reduced by the control latency to stay stable, and
                # a capped drag speed so acceleration stays moderate.
                gain = env.bridge.effective_gain()
                k = DRAG_GAIN / (1 + env.dynamics.action_delay)
            else:
                gain, k = env.bridge.gain, DRAG_GAIN
            bx, by = k * np.array([-err[1], -err[0]]) / gain
            c, s = np.cos(env.mouse_yaw()), np.sin(env.mouse_yaw())
            step = np.array([c * bx - s * by, s * bx + c * by])
            if env.realistic and np.linalg.norm(step) > REALISTIC_DRAG_MAX:
                step *= REALISTIC_DRAG_MAX / np.linalg.norm(step)
            self.waypoint = np.array([*(self._target[:2] + step), press_z])
        else:
            self.waypoint = np.array([*self._target[:2], press_z])
