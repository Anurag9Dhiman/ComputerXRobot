"""Procedural scene variation: the generalisation axes of the benchmark.

A `SceneVariation` fixes everything about the scene that is not the task: where the keyboard
and mouse sit on the desk, lighting, colours, the desktop wallpaper, a small robot-camera
offset, the mouse's cursor gain (DPI), keycap stiffness and mouse friction. Variations are
drawn from named distributions:

    none          the canonical scene
    id            the training distribution: modest changes on every axis
    ood_layout    keyboard and mouse moved beyond the training range (everything else id)
    ood_visual    lighting, colours, wallpaper and camera offset beyond the training range
    ood_dynamics  cursor gain, key stiffness and mouse friction beyond the training range

Each out-of-distribution (OOD) range is disjoint from the id range on its axis, so the OOD
test splits measure generalisation rather than interpolation. Every variation in every
distribution is solvable: the scripted oracle succeeds on all of them (tests/test_benchmark.py).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from environment.computer_env import ComputerUseEnv

DISTRIBUTIONS = ("none", "id", "ood_layout", "ood_visual", "ood_dynamics")

NOMINAL_DESK = (0.66, 0.49, 0.33)
NOMINAL_WALLPAPER = (44, 82, 138)
NOMINAL_GAIN = 6400.0

ID_DESKS = [(0.66, 0.49, 0.33), (0.58, 0.42, 0.28), (0.72, 0.56, 0.40), (0.62, 0.50, 0.38)]
OOD_DESKS = [(0.36, 0.36, 0.39), (0.86, 0.86, 0.83), (0.24, 0.17, 0.12), (0.45, 0.55, 0.42)]
ID_WALLPAPERS = [(44, 82, 138), (38, 70, 120), (52, 92, 150), (40, 100, 140)]
OOD_WALLPAPERS = [(30, 92, 60), (92, 50, 122), (70, 70, 76), (160, 92, 40)]


@dataclass(frozen=True)
class SceneVariation:
    keyboard_offset: tuple[float, float] = (0.0, 0.0)  # (dx, dy) on the desk, m
    mouse_offset: tuple[float, float] = (0.0, 0.0)  # mouse pad and mouse, m
    light_scale: float = 1.0  # key-light intensity multiplier
    light_azimuth: float = 0.0  # key-light direction rotated about z, rad
    desk_rgb: tuple[float, float, float] = NOMINAL_DESK
    floor_scale: float = 1.0  # floor brightness multiplier
    wallpaper_rgb: tuple[int, int, int] = NOMINAL_WALLPAPER
    camera_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)  # robot_cam position, m
    cursor_gain: float = NOMINAL_GAIN  # px per metre of mouse travel
    key_stiffness_scale: float = 1.0
    mouse_friction_scale: float = 1.0
    name: str = field(default="none", compare=False)  # distribution it was drawn from

    def to_dict(self) -> dict:
        return asdict(self)


NOMINAL = SceneVariation()


def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def _signed(rng, lo, hi):
    """|x| in [lo, hi] with a random sign."""
    return float(rng.choice([-1.0, 1.0]) * rng.uniform(lo, hi))


def _id_layout(rng):
    return dict(keyboard_offset=(_u(rng, -0.02, 0.02), _u(rng, -0.025, 0.03)),
                mouse_offset=(_u(rng, -0.02, 0.02), _u(rng, -0.03, 0.02)))


def _id_visual(rng):
    return dict(light_scale=_u(rng, 0.8, 1.2), light_azimuth=_u(rng, -0.3, 0.3),
                desk_rgb=ID_DESKS[rng.integers(len(ID_DESKS))], floor_scale=_u(rng, 0.85, 1.15),
                wallpaper_rgb=ID_WALLPAPERS[rng.integers(len(ID_WALLPAPERS))],
                camera_offset=tuple(rng.uniform(-0.008, 0.008, 3).tolist()))


def _id_dynamics(rng):
    return dict(cursor_gain=_u(rng, 5600, 7200), key_stiffness_scale=_u(rng, 0.9, 1.1),
                mouse_friction_scale=_u(rng, 0.8, 1.2))


def sample_variation(distribution: str, rng: np.random.Generator) -> SceneVariation:
    """Draw a scene from a named distribution (see module docstring)."""
    if distribution == "none":
        return NOMINAL
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"unknown variation distribution {distribution!r}; choose from {DISTRIBUTIONS}")
    params = {**_id_layout(rng), **_id_visual(rng), **_id_dynamics(rng)}
    if distribution == "ood_layout":
        # Keyboard further to the robot's left and the mouse further right, both shifted in depth.
        params.update(keyboard_offset=(_signed(rng, 0.03, 0.045), _u(rng, 0.04, 0.06)),
                      mouse_offset=(_signed(rng, 0.03, 0.045), _u(rng, -0.07, -0.04)))
    elif distribution == "ood_visual":
        params.update(light_scale=float(rng.choice([_u(rng, 0.45, 0.6), _u(rng, 1.5, 1.7)])),
                      light_azimuth=_signed(rng, 0.6, 1.0),
                      desk_rgb=OOD_DESKS[rng.integers(len(OOD_DESKS))],
                      floor_scale=float(rng.choice([_u(rng, 0.5, 0.65), _u(rng, 1.4, 1.6)])),
                      wallpaper_rgb=OOD_WALLPAPERS[rng.integers(len(OOD_WALLPAPERS))],
                      camera_offset=tuple((rng.choice([-1.0, 1.0], 3) * rng.uniform(0.015, 0.025, 3)).tolist()))
    elif distribution == "ood_dynamics":
        params.update(cursor_gain=float(rng.choice([_u(rng, 4000, 4800), _u(rng, 8400, 9600)])),
                      key_stiffness_scale=_u(rng, 1.3, 1.5),
                      mouse_friction_scale=_u(rng, 1.6, 2.0))
    return SceneVariation(**params, name=distribution)


class VariationApplier:
    """Writes a SceneVariation into the MuJoCo model (and desktop / mouse mapping) of an env,
    always starting from the canonical values, so variations never accumulate."""

    def __init__(self, env: "ComputerUseEnv"):
        m = env.model
        self.env = env
        self.ids = {
            "keyboard": m.body("keyboard").id,
            "mouse": m.body("mouse").id,
            "mouse_pad": m.geom("mouse_pad").id,
            "keyboard_cam": m.camera("keyboard_cam").id,
            "mouse_cam": m.camera("mouse_cam").id,
            "robot_cam": m.camera("robot_cam").id,
            "key_light": m.light("key_light").id,
            "desk": m.material("desk").id,
            "floor": m.material("floor").id,
        }
        from environment.keyboard_layout import KEYS

        self.key_joints = np.array([m.joint(k.joint).id for k in KEYS])
        self.mouse_dofs = np.array([m.joint(n).dofadr[0] for n in ("mouse_x", "mouse_y")])
        self.nominal = {
            "body_pos": m.body_pos.copy(), "geom_pos": m.geom_pos.copy(), "cam_pos": m.cam_pos.copy(),
            "light_diffuse": m.light_diffuse.copy(), "light_dir": m.light_dir.copy(),
            "mat_rgba": m.mat_rgba.copy(), "jnt_stiffness": m.jnt_stiffness.copy(),
            "dof_frictionloss": m.dof_frictionloss.copy(),
        }

    def apply(self, var: SceneVariation) -> None:
        m, n, i = self.env.model, self.nominal, self.ids
        for key in ("body_pos", "geom_pos", "cam_pos", "light_diffuse", "light_dir", "mat_rgba",
                    "jnt_stiffness", "dof_frictionloss"):
            getattr(m, key)[:] = n[key]

        kb = np.array([*var.keyboard_offset, 0.0])
        ms = np.array([*var.mouse_offset, 0.0])
        m.body_pos[i["keyboard"]] += kb
        m.cam_pos[i["keyboard_cam"]] += kb
        m.body_pos[i["mouse"]] += ms
        m.geom_pos[i["mouse_pad"]] += ms
        m.cam_pos[i["mouse_cam"]] += ms
        m.cam_pos[i["robot_cam"]] += np.asarray(var.camera_offset)

        light = i["key_light"]
        m.light_diffuse[light] = np.clip(n["light_diffuse"][light] * var.light_scale, 0, 1)
        c, s = np.cos(var.light_azimuth), np.sin(var.light_azimuth)
        dx, dy, dz = n["light_dir"][light]
        m.light_dir[light] = (c * dx - s * dy, s * dx + c * dy, dz)
        m.mat_rgba[i["desk"], :3] = var.desk_rgb
        m.mat_rgba[i["floor"], :3] = np.clip(n["mat_rgba"][i["floor"], :3] * var.floor_scale, 0, 1)

        m.jnt_stiffness[self.key_joints] = n["jnt_stiffness"][self.key_joints] * var.key_stiffness_scale
        m.dof_frictionloss[self.mouse_dofs] = n["dof_frictionloss"][self.mouse_dofs] * var.mouse_friction_scale

        self.env.bridge.gain = var.cursor_gain
        self.env.desktop.set_wallpaper(var.wallpaper_rgb)
