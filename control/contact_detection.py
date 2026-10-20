"""Detect key presses, mouse-button presses, mouse motion and robot contacts.

Key and button presses are read from the spring-loaded slide joints with hysteresis: a
key goes down when it travels past PRESS_FRACTION of its stroke and comes back up only
when it returns above RELEASE_FRACTION. This keeps a resting or bouncing finger from
producing repeated events.

Contacts are classified so evaluation can count:
  * collisions       robot touching something it should not (desk, monitor, keyboard plate,
                     keys touched by the palm or arm, ...), counted per contact onset;
  * failed contacts  finger touched a key or the mouse button but released it without the
                     press registering.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from environment.keyboard_layout import KEYS

PRESS_FRACTION = 0.6
RELEASE_FRACTION = 0.3

ROBOT_ROOT_BODY = "link_yaw"
FINGER_GEOM = "finger"
PALM_GEOMS = ("palm", "finger_housing")
MOUSE_SHELL_GEOMS = ("mouse_rest", "mouse_base")
MOUSE_BUTTON_JOINT = "mouse_button"
MOUSE_BUTTON_GEOM = "mouse_button"


@dataclass(frozen=True)
class DeviceEvent:
    kind: str  # "key_down" | "key_up" | "button_down" | "button_up"
    name: str  # key name (e.g. "a", "enter") or "mouse_button"
    char: str | None = None


class DeviceMonitor:
    """Turns physical device state into discrete events and contact statistics."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        key_joints = [model.joint(k.joint) for k in KEYS]
        self.key_qadr = np.array([j.qposadr[0] for j in key_joints])
        travel = np.array([-model.jnt_range[j.id][0] for j in key_joints])
        self.key_press_q = -PRESS_FRACTION * travel
        self.key_release_q = -RELEASE_FRACTION * travel

        button = model.joint(MOUSE_BUTTON_JOINT)
        self.button_qadr = button.qposadr[0]
        button_travel = -model.jnt_range[button.id][0]
        self.button_press_q = -PRESS_FRACTION * button_travel
        self.button_release_q = -RELEASE_FRACTION * button_travel

        self.mouse_qadr = np.array([model.joint(n).qposadr[0] for n in ("mouse_x", "mouse_y")])
        self.mouse_range = np.array([model.jnt_range[model.joint(n).id] for n in ("mouse_x", "mouse_y")])

        # Geom classification for contact analysis.
        root = model.body(ROBOT_ROOT_BODY).id
        self.robot_geoms = {
            g for g in range(model.ngeom) if self._is_descendant(model.geom_bodyid[g], root)
        }
        self.finger_geom = model.geom(FINGER_GEOM).id
        self.palm_geoms = {model.geom(n).id for n in PALM_GEOMS}
        self.key_geom_to_index = {model.geom(k.body).id: i for i, k in enumerate(KEYS)}
        self.mouse_shell_geoms = {model.geom(n).id for n in MOUSE_SHELL_GEOMS}
        self.button_geom = model.geom(MOUSE_BUTTON_GEOM).id
        self.reset()

    def _is_descendant(self, body: int, root: int) -> bool:
        while body != 0:
            if body == root:
                return True
            body = self.model.body_parentid[body]
        return False

    # ---------------------------------------------------------------- state
    def reset(self, data: mujoco.MjData | None = None) -> None:
        self.key_down = np.zeros(len(KEYS), dtype=bool)
        self.button_down = False
        self.press_count = 0
        self.click_count = 0
        self.collision_count = 0
        self.failed_contact_count = 0
        self.palm_on_mouse = False
        self._active_pairs: set[tuple[int, int]] = set()
        # Finger contact episodes: target ("key", i) / ("button",) -> pressed during contact?
        self._finger_episodes: dict[tuple, bool] = {}
        if data is not None:
            self.update(data)
            self.press_count = self.click_count = 0

    def mouse_position(self, data: mujoco.MjData) -> np.ndarray:
        """Mouse displacement (x, y) from its centre on the pad, metres."""
        return data.qpos[self.mouse_qadr].copy()

    def keys_down(self) -> list[str]:
        return [KEYS[i].name for i in np.flatnonzero(self.key_down)]

    # --------------------------------------------------------------- events
    def update(self, data: mujoco.MjData) -> list[DeviceEvent]:
        """Poll devices (call every physics step). Returns new press/release events."""
        events: list[DeviceEvent] = []
        q = data.qpos[self.key_qadr]
        went_down = ~self.key_down & (q < self.key_press_q)
        went_up = self.key_down & (q > self.key_release_q)
        for i in np.flatnonzero(went_down):
            events.append(DeviceEvent("key_down", KEYS[i].name, KEYS[i].char))
        for i in np.flatnonzero(went_up):
            events.append(DeviceEvent("key_up", KEYS[i].name, KEYS[i].char))
        self.key_down = (self.key_down | went_down) & ~went_up
        self.press_count += int(went_down.sum())

        qb = data.qpos[self.button_qadr]
        if not self.button_down and qb < self.button_press_q:
            self.button_down = True
            self.click_count += 1
            events.append(DeviceEvent("button_down", "mouse_button"))
        elif self.button_down and qb > self.button_release_q:
            self.button_down = False
            events.append(DeviceEvent("button_up", "mouse_button"))

        self._update_contacts(data)
        return events

    def _update_contacts(self, data: mujoco.MjData) -> None:
        pairs: set[tuple[int, int]] = set()
        finger_targets: set[tuple] = set()
        palm_on_mouse = False
        for c in data.contact[: data.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if g2 in self.robot_geoms:
                g1, g2 = g2, g1
            if g1 not in self.robot_geoms:
                continue
            if g1 == self.finger_geom and g2 in self.key_geom_to_index:
                finger_targets.add(("key", self.key_geom_to_index[g2]))
                continue
            if g1 == self.finger_geom and g2 == self.button_geom:
                finger_targets.add(("button",))
                continue
            if g1 in self.palm_geoms and g2 in self.mouse_shell_geoms:
                palm_on_mouse = True
                continue
            pairs.add((g1, g2))

        self.collision_count += len(pairs - self._active_pairs)
        self._active_pairs = pairs
        self.palm_on_mouse = palm_on_mouse

        for target in finger_targets:
            pressed = self.key_down[target[1]] if target[0] == "key" else self.button_down
            self._finger_episodes[target] = self._finger_episodes.get(target, False) or bool(pressed)
        for target in list(self._finger_episodes):
            if target not in finger_targets:
                if not self._finger_episodes.pop(target):
                    self.failed_contact_count += 1

    def collision_names(self) -> list[tuple[str, str]]:
        """Names of the geom pairs currently counted as collisions (for debugging)."""
        name = lambda g: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g)
        return [(name(a), name(b)) for a, b in sorted(self._active_pairs)]
