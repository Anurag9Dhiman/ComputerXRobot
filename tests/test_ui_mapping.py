"""Physical device state -> UI events, without the robot."""

import mujoco

from control.contact_detection import DeviceMonitor
from environment.browser import Desktop
from environment.keyboard_layout import KEYS
from environment.ui_events import CURSOR_GAIN, PhysicalUIBridge


def _bridge(env):
    data = mujoco.MjData(env.model)
    mujoco.mj_forward(env.model, data)
    desktop = Desktop()
    desktop.reset(active="editor")
    bridge = PhysicalUIBridge(DeviceMonitor(env.model), desktop)
    bridge.reset(data)
    return data, desktop, bridge


def test_key_travel_produces_one_key_event_with_hysteresis(env):
    data, desktop, bridge = _bridge(env)
    adr = env.model.joint("key_h").qposadr[0]

    data.qpos[adr] = -0.0035  # past the actuation point
    events = bridge.update(data)
    assert [(e.kind, e.value) for e in events] == [("key", "h")]
    data.qpos[adr] = -0.002  # partly released: still down, no repeat
    assert bridge.update(data) == []
    data.qpos[adr] = 0.0  # released
    bridge.update(data)
    data.qpos[adr] = -0.0035  # second press
    bridge.update(data)
    assert desktop.apps["editor"].text == "hh"


def test_every_key_maps_to_its_character(env):
    data, desktop, bridge = _bridge(env)
    typed = ""
    for key in KEYS:
        adr = env.model.joint(key.joint).qposadr[0]
        data.qpos[adr] = -0.0035
        bridge.update(data)
        data.qpos[adr] = 0.0
        bridge.update(data)
        if key.char == "\b":
            typed = typed[:-1]
        else:
            typed += key.char
    assert desktop.apps["editor"].text == typed


def test_mouse_motion_moves_cursor_relative(env):
    data, desktop, bridge = _bridge(env)
    x, y = (env.model.joint(n).qposadr[0] for n in ("mouse_x", "mouse_y"))
    start = desktop.cursor
    data.qpos[y] -= 0.01  # 1 cm to the robot's right -> cursor right
    data.qpos[x] += 0.005  # 5 mm away from the robot -> cursor up
    bridge.update(data)
    assert desktop.cursor == (round(start[0] + CURSOR_GAIN * 0.01), round(start[1] - CURSOR_GAIN * 0.005))


def test_left_button_clicks_widget_under_cursor(env):
    data, desktop, bridge = _bridge(env)
    desktop.reset(active=None)
    desktop.set_cursor(desktop.widget_rect("dock:browser").center)
    data.qpos[env.model.joint("mouse_left").qposadr[0]] = -0.0035
    events = bridge.update(data)
    assert [(e.kind, e.value) for e in events] == [("click", "dock:browser")]
    assert desktop.active == "browser"


def test_right_button_opens_context_menu(env):
    data, desktop, bridge = _bridge(env)
    desktop.apps["editor"].text = "some text"
    desktop.set_cursor(desktop.widget_rect("editor:text").center)
    right = env.model.joint("mouse_right").qposadr[0]
    data.qpos[right] = -0.0035
    events = bridge.update(data)
    assert [(e.kind, e.value, e.accepted) for e in events] == [("click", "right:editor:text", True)]
    assert desktop.state()["menu"] == ["menu:clear", "menu:close"]
    data.qpos[right] = 0.0
    bridge.update(data)

    desktop.set_cursor(desktop.widget_rect("menu:clear").center)  # then left-click "Clear all"
    data.qpos[env.model.joint("mouse_left").qposadr[0]] = -0.0035
    events = bridge.update(data)
    assert [(e.kind, e.value) for e in events] == [("click", "menu:clear")]
    assert desktop.apps["editor"].text == "" and desktop.menu is None
