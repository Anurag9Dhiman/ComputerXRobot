"""Ground truth for recognising every part of the scene in a camera image.

`annotate(env, camera, height, width)` projects the scene's components through a named
camera and returns one record per component in view:

    kind      "key" | "mouse_button" | "mouse_wheel" | "mouse" | "keyboard" | "screen" |
              "screen_text" | "cursor"
    name      e.g. "key_a", "key_enter", "mouse_left", "screen"
    label     what is written on it or what it is: "A", "Enter", "L", "R", the screen text
    bbox      (x0, y0, x1, y1) image pixels, clipped to the image
    visible   fraction of the component's projected area the camera actually sees, measured
              with a segmentation render (0 when hidden, e.g. under the robot's hand)

and kind-specific fields: `char` for keys, `button` for mouse buttons, `screen_bbox` (screen
pixels) for screen text. Together with the RGB image this is supervision for detecting keys,
reading their legends, telling the left and right mouse buttons apart, and reading the screen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from environment.browser import SCREEN_H, SCREEN_W
from environment.keyboard_layout import CAP_CENTER_Z, CAP_HALF_HEIGHT, KEYS

if TYPE_CHECKING:
    from environment.computer_env import ComputerUseEnv

# Screen quad corners in the monitor body frame (must match assets/desk.xml "screen_quad").
SCREEN_X, SCREEN_HALF_W, SCREEN_HALF_H = -0.0205, 0.24, 0.15
MOUSE_BUTTON_HALF = (0.015, 0.01375, 0.004)
MOUSE_BASE_HALF = (0.055, 0.032)
MOUSE_TOP_Z = 0.026
CURSOR_SIZE_PX = (13, 21)


def project(env: "ComputerUseEnv", camera: str, points: np.ndarray, height: int, width: int):
    """World points (N, 3) -> image pixels (N, 2) and depth along the view axis (N,)."""
    cam = env.model.camera(camera).id
    pos = env.data.cam_xpos[cam]
    rot = env.data.cam_xmat[cam].reshape(3, 3)  # columns: camera x (right), y (up), z (back)
    local = (np.asarray(points, float) - pos) @ rot
    depth = -local[:, 2]
    f = 0.5 * height / np.tan(np.deg2rad(env.model.cam_fovy[cam]) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = width / 2 + f * local[:, 0] / depth
        v = height / 2 - f * local[:, 1] / depth
    return np.stack([u, v], axis=1), depth


def _box_corners(half: tuple[float, ...], z0: float, z1: float) -> np.ndarray:
    hx, hy = half[0], half[1]
    return np.array([[sx * hx, sy * hy, z] for sx in (-1, 1) for sy in (-1, 1) for z in (z0, z1)])


def _hull_area(pts: np.ndarray) -> float:
    """Area of the convex hull of 2-D points (Andrew's monotone chain + shoelace)."""
    pts = sorted(map(tuple, pts))
    if len(pts) < 3:
        return 0.0

    def half(seq):
        out: list = []
        for p in seq:
            while len(out) >= 2 and (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1]) - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0]) <= 0:
                out.pop()
            out.append(p)
        return out

    hull = half(pts)[:-1] + half(pts[::-1])[:-1]
    x, y = np.array(hull).T
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


class _Annotator:
    def __init__(self, env: "ComputerUseEnv", camera: str, height: int, width: int):
        self.env, self.camera, self.h, self.w = env, camera, height, width
        self.seg = env.segmentation(camera, height, width)
        self.records: list[dict] = []

    def body_points(self, body: str, local: np.ndarray) -> np.ndarray:
        b = self.env.model.body(body).id
        return self.env.data.xpos[b] + local @ self.env.data.xmat[b].reshape(3, 3).T

    def add(self, kind: str, name: str, label: str, world: np.ndarray, geoms: set[int], **extra) -> None:
        uv, depth = project(self.env, self.camera, world, self.h, self.w)
        if np.any(depth <= 0):
            return
        # Box edges are pixel boundaries; round to the nearest one (floor/ceil would grow a box
        # that is exactly pixel-aligned, e.g. in screen_cam, by a pixel on each side).
        x0, y0 = np.round(uv.min(axis=0)).astype(int)
        x1, y1 = np.round(uv.max(axis=0)).astype(int)
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, self.w), min(y1, self.h)
        if cx0 >= cx1 or cy0 >= cy1:
            return
        area = max(_hull_area(uv), 1.0)
        crop = self.seg[cy0:cy1, cx0:cx1]
        visible = float(np.isin(crop, list(geoms)).sum()) / area
        self.records.append({
            "kind": kind, "name": name, "label": label,
            "bbox": (int(cx0), int(cy0), int(cx1), int(cy1)),
            "visible": round(min(visible, 1.0), 3),
            **extra,
        })

    def geoms(self, *names: str) -> set[int]:
        return {self.env.model.geom(n).id for n in names}


def annotate(env: "ComputerUseEnv", camera: str, height: int, width: int) -> list[dict]:
    a = _Annotator(env, camera, height, width)
    model = env.model

    # Keyboard and keys: the top face of each keycap carries its legend.
    key_geoms = set()
    top = CAP_CENTER_Z + CAP_HALF_HEIGHT
    for k in KEYS:
        hx, hy = k.cap_half_size
        corners = np.array([[sx * hx, sy * hy, top] for sx in (-1, 1) for sy in (-1, 1)])
        geoms = a.geoms(k.body, f"label_{k.name}")
        key_geoms |= geoms
        a.add("key", k.body, k.label or k.name, a.body_points(k.body, corners), geoms, char=k.char)
    plate = model.geom("keyboard_plate")
    a.add("keyboard", "keyboard", "keyboard",
          a.body_points("keyboard", _box_corners(plate.size, 0.0, 2 * plate.size[2] + top)),
          key_geoms | {plate.id})

    # Mouse: shell, the two labelled buttons and the wheel.
    mouse_geoms = a.geoms("mouse_base", "mouse_rest", "mouse_wheel")
    for button, label in (("left", "L"), ("right", "R")):
        body = f"mouse_{button}"
        geoms = a.geoms(body, f"label_{body}")
        mouse_geoms |= geoms
        hx, hy, hz = MOUSE_BUTTON_HALF
        corners = np.array([[sx * hx, sy * hy, hz] for sx in (-1, 1) for sy in (-1, 1)])
        a.add("mouse_button", body, label, a.body_points(body, corners), geoms, button=button)
    wheel = model.geom("mouse_wheel")
    a.add("mouse_wheel", "mouse_wheel", "wheel",
          env.data.geom_xpos[wheel.id] + _box_corners((wheel.size[0], wheel.size[1]), -wheel.size[0], wheel.size[0]),
          {wheel.id})
    a.add("mouse", "mouse", "mouse", a.body_points("mouse", _box_corners(MOUSE_BASE_HALF, 0.0, MOUSE_TOP_Z)),
          mouse_geoms)

    # Monitor screen, the text drawn on it, and the cursor.
    screen_geom = a.geoms("screen")

    def screen_points(px: np.ndarray) -> np.ndarray:
        """Screen pixel coordinates (N, 2) -> world points on the screen quad."""
        local = np.stack([
            np.full(len(px), SCREEN_X),
            SCREEN_HALF_W - px[:, 0] / SCREEN_W * 2 * SCREEN_HALF_W,
            SCREEN_HALF_H - px[:, 1] / SCREEN_H * 2 * SCREEN_HALF_H,
        ], axis=1)
        return a.body_points("monitor", local)

    def rect_corners(x0, y0, x1, y1) -> np.ndarray:
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)

    a.add("screen", "screen", "screen", screen_points(rect_corners(0, 0, SCREEN_W, SCREEN_H)), screen_geom)
    env.desktop.render()  # make sure text_items describe what is on the monitor
    for i, item in enumerate(env.desktop.text_items):
        a.add("screen_text", f"screen_text_{i}", item["text"], screen_points(rect_corners(*item["bbox"])),
              screen_geom, screen_bbox=item["bbox"])
    cx, cy = env.desktop.cursor
    a.add("cursor", "cursor", "cursor",
          screen_points(rect_corners(cx, cy, cx + CURSOR_SIZE_PX[0], cy + CURSOR_SIZE_PX[1])), screen_geom)
    return a.records


ANNOTATION_COLORS = {"key": (255, 140, 0), "mouse_button": (0, 200, 255), "mouse_wheel": (180, 80, 255),
                     "screen_text": (40, 220, 90), "cursor": (255, 40, 160)}
HIGHLIGHT_COLOR = (60, 255, 60)


def draw_annotations(image: np.ndarray, records: list[dict], min_visible: float = 0.5,
                     tags: tuple[str, ...] = ("key", "mouse_button"),
                     highlight: dict[str, str] | None = None) -> np.ndarray:
    """Copy of `image` with each sufficiently visible record's box drawn on it.

    tags       kinds whose label is printed above the box
    highlight  {record name: caption} drawn with a thick bright box, e.g. the key just pressed
    """
    import pygame

    from environment.browser import Fonts

    highlight = highlight or {}
    surf = pygame.surfarray.make_surface(np.ascontiguousarray(image.transpose(1, 0, 2)))
    font = Fonts.get(14)
    for r in records:
        if r["kind"] not in ANNOTATION_COLORS or r["visible"] < min_visible or r["name"] in highlight:
            continue
        x0, y0, x1, y1 = r["bbox"]
        color = ANNOTATION_COLORS[r["kind"]]
        pygame.draw.rect(surf, color, (x0, y0, x1 - x0, y1 - y0), 1)
        if r["kind"] in tags:
            tag = font.render(r["label"], True, (0, 0, 0), color)
            surf.blit(tag, (x0, max(0, y0 - tag.get_height())))
    big = Fonts.get(20)
    for r in records:
        if r["name"] in highlight:
            x0, y0, x1, y1 = r["bbox"]
            pygame.draw.rect(surf, HIGHLIGHT_COLOR, (x0 - 2, y0 - 2, x1 - x0 + 4, y1 - y0 + 4), 3)
            tag = big.render(highlight[r["name"]], True, (0, 0, 0), HIGHLIGHT_COLOR)
            surf.blit(tag, (min(x0, surf.get_width() - tag.get_width()), max(0, y0 - tag.get_height() - 2)))
    return pygame.surfarray.array3d(surf).transpose(1, 0, 2)
