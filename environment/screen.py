"""Bridge between pygame surfaces and MuJoCo textures, plus an offscreen renderer that
keeps the GPU copies of those textures up to date.

* `ScreenTexture` copies the desktop surface into the monitor's "screen" texture.
* `paint_keycap_labels` draws the key legends into the keyboard's label atlas.
* `paint_mouse_labels` draws "L" / "R" onto the mouse buttons.
* `SceneRenderer` renders any camera (RGB or segmentation) and re-uploads the screen texture
  when it changed.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pygame

from environment.browser import Fonts
from environment.keyboard_layout import ATLAS_HEIGHT, ATLAS_WIDTH, CAP_RGB, KEYS

MOUSE_BUTTON_RGB = (0.97, 0.97, 0.97)
MOUSE_LEGEND_RGB = (0.06, 0.12, 0.32)
MOUSE_LABELS = ("L", "R")  # left / right halves of the "mouselabels" texture


def _surface_to_rgb(surface: pygame.Surface) -> np.ndarray:
    """(H, W, 3) uint8 array; row 0 is the top of the surface."""
    return pygame.surfarray.array3d(surface).transpose(1, 0, 2)


class _Texture:
    def __init__(self, model: mujoco.MjModel, name: str):
        self.model = model
        self.id = model.texture(name).id
        self.width = int(model.tex_width[self.id])
        self.height = int(model.tex_height[self.id])
        if model.tex_nchannel[self.id] != 3:
            raise ValueError(f"texture {name!r} must be RGB")
        adr = int(model.tex_adr[self.id])
        self._view = model.tex_data[adr : adr + self.width * self.height * 3]

    def write_rgb(self, rgb: np.ndarray) -> None:
        if rgb.shape != (self.height, self.width, 3):
            raise ValueError(f"expected {(self.height, self.width, 3)}, got {rgb.shape}")
        self._view[:] = rgb.reshape(-1)


class ScreenTexture(_Texture):
    """The monitor's screen. `version` increases each time new pixels are written."""

    def __init__(self, model: mujoco.MjModel, name: str = "screen"):
        super().__init__(model, name)
        self.version = 0

    def write_surface(self, surface: pygame.Surface) -> None:
        if surface.get_size() != (self.width, self.height):
            surface = pygame.transform.smoothscale(surface, (self.width, self.height))
        self.write_rgb(_surface_to_rgb(surface))
        self.version += 1


def _rgb255(rgb) -> tuple[int, int, int]:
    return tuple(int(round(255 * c)) for c in rgb)


def _fitted_legend(text: str, box: pygame.Rect, color, max_height: float) -> pygame.Surface:
    """Render `text` as large as fits in `box`: glyph height up to max_height * box height."""
    size = int(box.height * max_height * 1.45)  # pygame font size -> cap height is ~0.7 * size
    while size > 8:
        w, h = Fonts.get(size).size(text)
        if w <= box.width * 0.88 and h * 0.7 <= box.height * max_height:
            break
        size -= 2
    return Fonts.get(size).render(text, True, color)


def label_image(text: str, size: tuple[int, int], background, color, max_height: float = 0.72) -> pygame.Surface:
    """A legend cell exactly as painted onto the keys / mouse buttons (also used as a reading
    template by the legibility tests)."""
    cell = pygame.Surface(size)
    cell.fill(_rgb255(background))
    edge = tuple(int(c * 0.75) for c in _rgb255(background))
    pygame.draw.rect(cell, edge, cell.get_rect(), max(2, size[1] // 40))
    if text:
        glyph = _fitted_legend(text, cell.get_rect(), _rgb255(color), max_height)
        cell.blit(glyph, glyph.get_rect(center=cell.get_rect().center))
    return cell


def paint_keycap_labels(model: mujoco.MjModel, name: str = "keylabels") -> None:
    tex = _Texture(model, name)
    assert (tex.width, tex.height) == (ATLAS_WIDTH, ATLAS_HEIGHT)
    atlas = pygame.Surface((tex.width, tex.height))
    atlas.fill(_rgb255(CAP_RGB))
    for key in KEYS:
        x0, y0, x1, y1 = key.atlas_rect()
        # Single characters fill most of the cap; words (Enter, Backspace, Space) are smaller.
        max_height = 0.72 if len(key.label) == 1 else 0.42
        atlas.blit(label_image(key.label, (x1 - x0, y1 - y0), key.cap_rgb, key.legend_rgb, max_height), (x0, y0))
    tex.write_rgb(_surface_to_rgb(atlas))


def paint_mouse_labels(model: mujoco.MjModel, name: str = "mouselabels") -> None:
    tex = _Texture(model, name)
    atlas = pygame.Surface((tex.width, tex.height))
    half = tex.width // len(MOUSE_LABELS)
    for i, text in enumerate(MOUSE_LABELS):
        atlas.blit(label_image(text, (half, tex.height), MOUSE_BUTTON_RGB, MOUSE_LEGEND_RGB, 0.6), (i * half, 0))
    tex.write_rgb(_surface_to_rgb(atlas))


class SceneRenderer:
    """Offscreen renderer for one resolution; renders any named camera."""

    def __init__(self, model: mujoco.MjModel, height: int, width: int, screen: ScreenTexture | None = None):
        self.model = model
        self.screen = screen
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self._uploaded_version = 0  # textures in the model at creation time are already on the GPU

    def _sync_textures(self) -> None:
        if self.screen is None or self.screen.version == self._uploaded_version:
            return
        gl_context = getattr(self.renderer, "_gl_context", None)
        if gl_context is not None:
            gl_context.make_current()
        mujoco.mjr_uploadTexture(self.model, self.renderer._mjr_context, self.screen.id)
        self._uploaded_version = self.screen.version

    def render(self, data: mujoco.MjData, camera: str | int) -> np.ndarray:
        self._sync_textures()
        self.renderer.update_scene(data, camera=camera)
        return self.renderer.render()

    def segmentation(self, data: mujoco.MjData, camera: str | int) -> np.ndarray:
        """(H, W) geom id per pixel, -1 where no geom is visible."""
        self.renderer.update_scene(data, camera=camera)
        self.renderer.enable_segmentation_rendering()
        try:
            seg = self.renderer.render()
        finally:
            self.renderer.disable_segmentation_rendering()
        ids = seg[..., 0].astype(np.int64)
        ids[seg[..., 1] != mujoco.mjtObj.mjOBJ_GEOM] = -1
        return ids

    def close(self) -> None:
        self.renderer.close()
