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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

from control.contact_detection import DeviceMonitor
from environment.browser import Desktop, click_target

CURSOR_GAIN = 6400.0  # pixels per metre of mouse travel (~10 cm across the 640 px screen)


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
    def __init__(self, devices: DeviceMonitor, desktop: Desktop, gain: float = CURSOR_GAIN):
        self.devices = devices
        self.desktop = desktop
        self.gain = gain
        self._last_mouse = np.zeros(2)

    def reset(self, data: mujoco.MjData) -> None:
        self.devices.reset(data)
        self._last_mouse = self.devices.mouse_position(data)

    def update(self, data: mujoco.MjData, expectation: Expectation | None = None) -> list[UIEvent]:
        """Poll the physical devices and apply the resulting events to the desktop."""
        device_events = self.devices.update(data)

        mouse = self.devices.mouse_position(data)
        dx, dy = mouse - self._last_mouse
        self._last_mouse = mouse
        if dx or dy:
            self.desktop.move_cursor(-self.gain * dy, -self.gain * dx)

        ui_events = []
        for ev in device_events:
            if ev.kind == "key_down":
                expected = expectation()[0] if expectation else None
                result = self.desktop.key_press(ev.char)
                ui_events.append(UIEvent(data.time, "key", ev.char, result.accepted, expected))
            elif ev.kind == "button_down":
                expected = expectation()[1] if expectation else None
                result = self.desktop.click(ev.name)
                accepted = result.target is not None or ev.name == "right"  # a right-click always opens a menu
                value = click_target(result.target, ev.name)
                ui_events.append(UIEvent(data.time, "click", value, accepted, expected))
        return ui_events
