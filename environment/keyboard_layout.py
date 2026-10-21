"""Keyboard layout and MJCF generator.

The layout is the single source of truth for the physical keyboard: key positions, key
names, the characters they produce, their legends and colours, and where each legend lives in
the keycap texture atlas. `assets/keyboard.xml` is generated from it:

    python -m environment.keyboard_layout          # rewrite assets/keyboard.xml

Coordinates: the robot sits at -x looking toward +x. The number row is farthest from the
robot (+x) and the space bar nearest. The robot's left is +y, so a row runs from +y to -y.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

KEY_PITCH = 0.019  # distance between neighbouring key centres (m), 1u
CAP_HALF = 0.008  # half size of a 1u keycap (m)
CAP_HALF_HEIGHT = 0.005
CAP_CENTER_Z = 0.012  # cap centre above the plate top (key body frame)
KEY_TRAVEL = 0.004  # m
KEY_STIFFNESS = 150.0  # N/m
KEY_PRELOAD_OFFSET = 0.003  # springref above the rest position -> ~0.45 N preload
PLATE_HALF_HEIGHT = 0.006
KEYBOARD_POS = (0.47, 0.06, 0.75)  # keyboard body origin (plate bottom, on the desk)

ATLAS_UNIT_PX = 128  # atlas pixels per 1u (a 1u legend is ~108 px square)
ATLAS_WIDTH = 2048
ATLAS_HEIGHT = 640  # 5 rows * 128 px

# High-contrast legends so a camera (and a model) can read every key: black on white
# keycaps, and white on dark slate for Enter / Backspace / Space so they stand out as a group.
CAP_RGB = (0.94, 0.94, 0.92)
SPECIAL_CAP_RGB = (0.22, 0.30, 0.42)
LEGEND_RGB = (0.04, 0.04, 0.06)
SPECIAL_LEGEND_RGB = (0.97, 0.97, 0.97)
PLATE_RGB = (0.07, 0.07, 0.08)

# (row stagger in units, [key, ...]); a key is a char or (name, width_in_units).
ROWS: list[tuple[float, list]] = [
    (0.0, list("1234567890-") + [("backspace", 2.0)]),
    (0.5, list("qwertyuiop")),
    (0.75, list("asdfghjkl;") + [("enter", 2.0)]),
    (1.25, list("zxcvbnm,./")),
    (3.25, [("space", 6.0)]),
]

_PUNCT_NAMES = {"-": "minus", ";": "semicolon", ",": "comma", ".": "period", "/": "slash"}
_SPECIAL_CHARS = {"backspace": "\b", "enter": "\n", "space": " "}
_SPECIAL_LABELS = {"backspace": "Backspace", "enter": "Enter", "space": "Space"}

TOTAL_WIDTH_UNITS = 13.0


@dataclass(frozen=True)
class KeySpec:
    name: str  # body/joint name suffix, e.g. "a", "minus", "enter"
    char: str  # character produced ("\n" for enter, "\b" for backspace)
    label: str  # text painted on the keycap
    row: int
    offset_units: float  # distance of the key's left edge from the keyboard's left edge
    width_units: float
    pos: tuple[float, float]  # key centre (x, y) in the keyboard frame

    @property
    def body(self) -> str:
        return f"key_{self.name}"

    @property
    def joint(self) -> str:
        return f"key_{self.name}"

    @property
    def special(self) -> bool:
        return self.name in _SPECIAL_CHARS

    @property
    def cap_rgb(self) -> tuple[float, float, float]:
        return SPECIAL_CAP_RGB if self.special else CAP_RGB

    @property
    def legend_rgb(self) -> tuple[float, float, float]:
        return SPECIAL_LEGEND_RGB if self.special else LEGEND_RGB

    @property
    def cap_half_size(self) -> tuple[float, float]:
        """Half extents (x, y) of the keycap."""
        return CAP_HALF, CAP_HALF + (self.width_units - 1.0) * KEY_PITCH / 2

    def atlas_rect(self) -> tuple[int, int, int, int]:
        """Pixel rectangle (x0, y0, x1, y1) of this key's label in the atlas texture."""
        hx, hy = self.cap_half_size
        cx = (self.offset_units + self.width_units / 2) * ATLAS_UNIT_PX
        cy = (self.row + 0.5) * ATLAS_UNIT_PX
        half_w = hy / KEY_PITCH * ATLAS_UNIT_PX  # label width runs along -y
        half_h = hx / KEY_PITCH * ATLAS_UNIT_PX
        return (round(cx - half_w), round(cy - half_h), round(cx + half_w), round(cy + half_h))


def _build_layout() -> list[KeySpec]:
    keys = []
    y_left = TOTAL_WIDTH_UNITS / 2 * KEY_PITCH
    for row, (stagger, row_keys) in enumerate(ROWS):
        offset = stagger
        for item in row_keys:
            if isinstance(item, tuple):
                name, width = item
                char, label = _SPECIAL_CHARS[name], _SPECIAL_LABELS[name]
            else:
                char, width = item, 1.0
                name = _PUNCT_NAMES.get(char, char)
                label = char.upper()
            x = (2 - row) * KEY_PITCH
            y = y_left - (offset + width / 2) * KEY_PITCH
            keys.append(KeySpec(name, char, label, row, offset, width, (x, y)))
            offset += width
    return keys


KEYS: list[KeySpec] = _build_layout()
KEY_BY_CHAR: dict[str, KeySpec] = {k.char: k for k in KEYS}
KEY_BY_NAME: dict[str, KeySpec] = {k.name: k for k in KEYS}
TYPEABLE_CHARS = "".join(k.char for k in KEYS if k.char not in "\b\n")


def _fmt(*values: float) -> str:
    return " ".join(f"{v:.5g}" for v in values)


def generate_keyboard_xml() -> str:
    """Return the MJCF for the keyboard (assets/keyboard.xml)."""
    plate_hx = 2.5 * KEY_PITCH + 0.010
    plate_hy = TOTAL_WIDTH_UNITS / 2 * KEY_PITCH + 0.010
    label_z = CAP_CENTER_Z + CAP_HALF_HEIGHT + 0.0003
    plate_top = 2 * PLATE_HALF_HEIGHT

    meshes, bodies = [], []
    for key in KEYS:
        hx, hy = key.cap_half_size
        x0, y0, x1, y1 = key.atlas_rect()
        u0, u1 = x0 / ATLAS_WIDTH, x1 / ATLAS_WIDTH
        v0, v1 = y0 / ATLAS_HEIGHT, y1 / ATLAS_HEIGHT
        # Label quad on the cap top, read by the robot: "up" is +x, "right" is -y.
        verts = _fmt(hx, hy, label_z, hx, -hy, label_z, -hx, -hy, label_z, -hx, hy, label_z)
        uvs = _fmt(u0, v0, u1, v0, u1, v1, u0, v1)
        meshes.append(
            f'    <mesh name="label_{key.name}" inertia="shell" vertex="{verts}" '
            f'texcoord="{uvs}" face="0 2 1  0 3 2"/>'
        )
        bodies.append(
            f'      <body name="{key.body}" pos="{_fmt(key.pos[0], key.pos[1], plate_top)}" childclass="key">\n'
            f'        <inertial pos="0 0 {CAP_CENTER_Z:g}" mass="0.005" diaginertia="1e-7 1e-7 1e-7"/>\n'
            f'        <joint name="{key.joint}"/>\n'
            f'        <geom name="{key.body}" size="{_fmt(hx, hy, CAP_HALF_HEIGHT)}" rgba="{_fmt(*key.cap_rgb)} 1"/>\n'
            f'        <geom name="label_{key.name}" class="key_label" mesh="label_{key.name}"/>\n'
            f"      </body>"
        )

    return f"""<mujoco model="cxr_keyboard">
  <!-- GENERATED by `python -m environment.keyboard_layout`. Do not edit by hand. -->
  <!-- {len(KEYS)} keys; each key is a body on a spring-loaded slide joint (travel {KEY_TRAVEL * 1000:.0f} mm). -->
  <default>
    <default class="key">
      <joint type="slide" axis="0 0 1" range="{_fmt(-KEY_TRAVEL)} 0" stiffness="{KEY_STIFFNESS:g}"
             springref="{KEY_PRELOAD_OFFSET:g}" damping="0.3" armature="0.002" solreflimit="0.005 1"/>
      <geom type="box" pos="0 0 {CAP_CENTER_Z:g}" contype="0" conaffinity="1" friction="0.8 0.02 0.001"/>
      <default class="key_label">
        <geom type="mesh" pos="0 0 0" material="keylabels" contype="0" conaffinity="0" group="1" rgba="1 1 1 1"/>
      </default>
    </default>
  </default>

  <asset>
    <!-- Legends are painted into this atlas at runtime (environment/screen.py). -->
    <texture name="keylabels" type="2d" builtin="flat" width="{ATLAS_WIDTH}" height="{ATLAS_HEIGHT}"
             rgb1="{_fmt(*CAP_RGB)}"/>
    <material name="keylabels" texture="keylabels" specular="0.1"/>
{chr(10).join(meshes)}
  </asset>

  <worldbody>
    <body name="keyboard" pos="{_fmt(*KEYBOARD_POS)}">
      <geom name="keyboard_plate" type="box" size="{_fmt(plate_hx, plate_hy, PLATE_HALF_HEIGHT)}"
            pos="0 0 {PLATE_HALF_HEIGHT:g}" contype="0" conaffinity="1" rgba="{_fmt(*PLATE_RGB)} 1"/>
{chr(10).join(bodies)}
    </body>
  </worldbody>
</mujoco>
"""


DEFAULT_PATH = Path(__file__).resolve().parent.parent / "assets" / "keyboard.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    args.out.write_text(generate_keyboard_xml())
    print(f"wrote {len(KEYS)} keys to {args.out}")


if __name__ == "__main__":
    main()
