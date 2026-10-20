"""Bridge between pygame surfaces and MuJoCo textures, plus an offscreen renderer that
keeps the GPU copies of those textures up to date.

* `ScreenTexture` copies the desktop surface into the monitor's "screen" texture.
* `paint_keycap_labels` draws the key legends into the keyboard's label atlas.
* `SceneRenderer` renders any camera and re-uploads the screen texture when it changed.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pygame

from environment.browser import Fonts
from environment.keyboard_layout import ATLAS_HEIGHT, ATLAS_WIDTH, KEYS


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


def paint_keycap_labels(model: mujoco.MjModel, name: str = "keylabels") -> None:
    tex = _Texture(model, name)
    assert (tex.width, tex.height) == (ATLAS_WIDTH, ATLAS_HEIGHT)
    atlas = pygame.Surface((tex.width, tex.height))
    cap = tuple(int(255 * c) for c in (0.16, 0.17, 0.19))
    atlas.fill(cap)
    for key in KEYS:
        x0, y0, x1, y1 = key.atlas_rect()
        rect = pygame.Rect(x0, y0, x1 - x0, y1 - y0)
        pygame.draw.rect(atlas, (60, 62, 68), rect, 2)
        if key.label:
            size = 44 if len(key.label) == 1 else 30
            img = Fonts.get(size).render(key.label, True, (235, 236, 240))
            atlas.blit(img, img.get_rect(center=rect.center))
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

    def close(self) -> None:
        self.renderer.close()
