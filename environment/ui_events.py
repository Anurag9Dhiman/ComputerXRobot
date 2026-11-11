"""Map physical interaction to UI events.

MuJoCo simulates the keyboard and mouse as rigid bodies; it knows nothing about an operating
system. This module is the explicit mapping between the two worlds:

    keyboard key travels past its actuation point  ->  key press   (desktop.key_press)
    mouse shell slides on the pad                  ->  cursor move (desktop.move_cursor)
    left mouse button past its actuation point     ->  click       (desktop.click("left"))
    right mouse button past its actuation point    ->  right-click (desktop.click("right"):
                                                       opens the context menu)

The mouse is relative, like a real one: cursor motion is proportional to how far the mouse
moved since the last update (CURSOR_GAIN pixels per metre) and the cursor is clamped at the
screen edges. Pushing the mouse away from the robot (+x) moves the cursor up; moving it to
the robot's right (-y) moves the cursor right.

In the realistic physics profile (environment/dynamics.py) the mouse can also rotate on the
pad. Its optical sensor measures motion in the mouse's own frame, so a rotated mouse moves
the cursor diagonally. With pointer acceleration the gain grows with mouse speed:
gain * (1 + accel * min(speed / ACCEL_REF_SPEED, 2)). Held keys auto-repeat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

from control.contact_detection import DeviceMonitor
from environment.browser import Desktop, click_target

CURSOR_GAIN = 6400.0  # pixels per metre of mouse travel (~10 cm across the 640 px screen)
ACCEL_REF_SPEED = 0.15  # m/s; pointer acceleration reaches 1x its extra gain at this mouse speed


@dataclass
class UIEvent:
    time: float
    kind: str  # "key" | "click"
    value: str | None  # character for keys; click target for clicks: widget id, "right:<widget>"
    # for right-clicks, None for a left click that hit nothing
    accepted: bool  # key reached a text field / click hit a widget
    expected: str | None = None  # what the task wanted at that moment (char or widget id)

    @property
    def correct(self) -> bool:
        """The event did exactly what the task wanted at that moment."""
        return self.accepted and self.value == self.expected


# Returns (expected_char, expected_click_target) for annotating events; either may be None.
Expectation = Callable[[], tuple[str | None, str | None]]


class PhysicalUIBridge:
    def __init__(self, devices: DeviceMonitor, desktop: Desktop, gain: float = CURSOR_GAIN,
                 yaw_qadr: int | None = None):
        self.devices = devices
        self.desktop = desktop
        self.gain = gain
        self.pointer_accel = 0.0
        self.yaw_qadr = yaw_qadr  # qpos address of the mouse's rotation joint (realistic profile)
        self._last_mouse = np.zeros(2)
        self._last_time = 0.0
        self._speed = 0.0

    def reset(self, data: mujoco.MjData) -> None:
        self.devices.reset(data)
        self._last_mouse = self.devices.mouse_position(data)
        self._last_time = data.time
        self._speed = 0.0

    @property
    def speed(self) -> float:
        """Smoothed mouse speed (m/s) that pointer acceleration responds to."""
        return self._speed

    def effective_gain(self, speed: float | None = None) -> float:
        """Pixels per metre at a mouse speed (default: the current speed)."""
        speed = self._speed if speed is None else speed
        return self.gain * (1.0 + self.pointer_accel * min(speed / ACCEL_REF_SPEED, 2.0))

    def cursor_delta(self, d_world: np.ndarray, yaw: float, speed: float) -> tuple[float, float]:
        """Cursor motion (du, dv) in pixels for a mouse displacement (dx, dy) in metres."""
        c, s = np.cos(yaw), np.sin(yaw)
        dx_b = c * d_world[0] + s * d_world[1]  # displacement in the mouse's own frame
        dy_b = -s * d_world[0] + c * d_world[1]
        gain = self.effective_gain(speed)
        return -gain * dy_b, -gain * dx_b

    def update(self, data: mujoco.MjData, expectation: Expectation | None = None) -> list[UIEvent]:
        """Poll the physical devices and apply the resulting events to the desktop."""
        device_events = self.devices.update(data)

        mouse = self.devices.mouse_position(data)
        d = mouse - self._last_mouse
        dt = data.time - self._last_time
        self._last_mouse, self._last_time = mouse, data.time
        if dt > 0:
            self._speed = 0.8 * self._speed + 0.2 * float(np.hypot(*d)) / dt  # ~10 ms smoothing
        if d.any():
            yaw = float(data.qpos[self.yaw_qadr]) if self.yaw_qadr is not None else 0.0
            self.desktop.move_cursor(*self.cursor_delta(d, yaw, self._speed))

        ui_events = []
        for ev in device_events:
            if ev.kind in ("key_down", "key_repeat"):
                expected = expectation()[0] if expectation else None
                result = self.desktop.key_press(ev.char)
                ui_events.append(UIEvent(data.time, "key", ev.char, result.accepted, expected))
            elif ev.kind == "button_down":
                expected = expectation()[1] if expectation else None
                result = self.desktop.click(ev.name, time=data.time)
                accepted = result.target is not None or ev.name == "right"  # a right-click always opens a menu
                value = click_target(result.target, ev.name)
                ui_events.append(UIEvent(data.time, "click", value, accepted, expected))
        return ui_events
