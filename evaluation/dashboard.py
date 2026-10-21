"""Demo views, used for the live window (`--display`), videos (`--video`) and GIFs (`--gif`).

`Dashboard` (--view dashboard): the 3D scene, the robot's camera, the simulated screen and
the task result in one frame.

    +------------------------------+--------------------+
    |                              |  robot camera      |
    |        3D scene (overview)   +--------------------+
    |                              |  simulated screen  |
    +------------------------------+----------+----+----+
    |  task / status / metrics     | keyboard |mouse|wrist|
    +------------------------------+----------+----+----+

The keyboard and mouse panels are the legible close-up cameras (keyboard_cam, mouse_cam).

`PerceptionView` (--view perception): what a model learns from. The close-up camera images
with their ground-truth labels (environment/annotations.py) drawn on top; the key or mouse
button just pressed is highlighted.

    +----------------------+--------------------------------+
    |  3D scene            |  screen_cam + text boxes        |
    +----------------------+                                |
    |  keyboard_cam + key  |                                |
    |  boxes               +----------------+---------------+
    +----------------------+  mouse_cam +   |  legend, task |
    |  task / events       |  L/R boxes     |  and counts   |
    +----------------------+----------------+---------------+
"""

from __future__ import annotations

import numpy as np
import pygame

from environment.annotations import ANNOTATION_COLORS, HIGHLIGHT_COLOR, draw_annotations
from environment.browser import Fonts, split_click
from environment.computer_env import ComputerUseEnv
from environment.keyboard_layout import KEY_BY_CHAR

WIDTH, HEIGHT = 1280, 720
MAIN = pygame.Rect(0, 0, 800, 500)
ROBOT_CAM = pygame.Rect(800, 0, 480, 300)
SCREEN = pygame.Rect(800, 300, 480, 300)
STATUS = pygame.Rect(0, 500, 800, 220)
KEYBOARD = pygame.Rect(800, 600, 240, 120)
MOUSE = pygame.Rect(1040, 600, 120, 120)
WRIST = pygame.Rect(1160, 600, 120, 120)

BG = (18, 20, 26)
FG = (232, 235, 240)
MUTED = (150, 156, 168)
GREEN = (70, 200, 110)
RED = (230, 90, 80)
AMBER = (240, 180, 60)


def _blit_array(surf: pygame.Surface, img: np.ndarray, rect: pygame.Rect) -> None:
    s = pygame.surfarray.make_surface(img.transpose(1, 0, 2))
    if s.get_size() != rect.size:
        s = pygame.transform.smoothscale(s, rect.size)
    surf.blit(s, rect)


def _label(surf, text, rect, size=20):
    font = Fonts.get(size)
    img = font.render(text, True, FG)
    bg = pygame.Surface((img.get_width() + 12, img.get_height() + 6), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 150))
    surf.blit(bg, (rect.x + 6, rect.y + 6))
    surf.blit(img, (rect.x + 12, rect.y + 9))


def _text(surf, text, pos, size=20, color=FG):
    surf.blit(Fonts.get(size).render(text, True, color), pos)


class Dashboard:
    def __init__(self, env: ComputerUseEnv, main_camera: str = "overview"):
        self.env = env
        self.main_camera = main_camera
        self.surface = pygame.Surface((WIDTH, HEIGHT))

    def draw(self, skill: str | None = None, policy_name: str = "") -> pygame.Surface:
        env, s = self.env, self.surface
        s.fill(BG)
        _blit_array(s, env.render(self.main_camera, MAIN.height, MAIN.width), MAIN)
        _label(s, "MuJoCo scene", MAIN)
        _blit_array(s, env.render("robot_cam", ROBOT_CAM.height, ROBOT_CAM.width), ROBOT_CAM)
        _label(s, "Robot camera (policy input)", ROBOT_CAM)
        _blit_array(s, env.screen_image(), SCREEN)
        _label(s, "Simulated screen", SCREEN)
        # Close-ups rendered at 2x and scaled down, so the legends stay crisp.
        for cam, rect, name in (("keyboard_cam", KEYBOARD, "Keyboard"), ("mouse_cam", MOUSE, "Mouse"),
                                ("wrist_cam", WRIST, "Wrist")):
            _blit_array(s, env.render(cam, 2 * rect.height, 2 * rect.width), rect)
            _label(s, name, rect, 16)
        self._draw_status(s, skill, policy_name)
        for r in (ROBOT_CAM, SCREEN, KEYBOARD, MOUSE, WRIST, STATUS):
            pygame.draw.rect(s, (40, 44, 54), r, 1)
        return s

    def frame(self, skill: str | None = None, policy_name: str = "") -> np.ndarray:
        return pygame.surfarray.array3d(self.draw(skill, policy_name)).transpose(1, 0, 2)

    def _draw_status(self, s, skill, policy_name):
        env = self.env
        m = env.metrics()
        x, y = STATUS.x + 16, STATUS.y + 12
        _text(s, env.task.instruction, (x, y), 28)
        y += 32
        sub = f"policy: {policy_name}" if policy_name else ""
        if skill:
            sub += f"   skill: {skill}"
        _text(s, f"t = {env.elapsed:5.2f} s   {sub}", (x, y), 20, MUTED)
        y += 28

        if env.status.success:
            badge, color = "SUCCESS", GREEN
        elif env.status.timeout:
            badge, color = "TIMEOUT", RED
        else:
            badge, color = "IN PROGRESS", AMBER
        pygame.draw.rect(s, color, (x, y, 150, 30), border_radius=6)
        _text(s, badge, (x + 12, y + 7), 24, (15, 15, 15))
        bar = pygame.Rect(x + 170, y + 8, 300, 14)
        pygame.draw.rect(s, (50, 54, 64), bar, border_radius=7)
        pygame.draw.rect(s, color, (bar.x, bar.y, int(bar.width * m["progress"]), bar.height), border_radius=7)
        _text(s, f"{100 * m['progress']:.0f}%", (bar.right + 10, y + 6), 20)
        y += 42

        state = env.desktop.state()
        if env.task.name in ("type_key", "type_text", "clear_text"):
            typed = state["editor_text"]
        else:
            typed = state["browser_query"]
        _text(s, f'typed: "{typed}"', (x, y), 22)
        y += 28
        _text(
            s,
            f"key presses {m['key_presses']}   clicks {m['clicks']}   wrong keys {m['wrong_keys']}   "
            f"failed clicks {m['failed_clicks']}   collisions {m['collisions']}   "
            f"failed contacts {m['failed_contacts']}",
            (x, y),
            19,
            MUTED,
        )
        y += 26
        events = env.tasks.events[-6:]
        log = "  ".join(
            (repr(e.value)[1:-1] if e.kind == "key" else f"click:{e.value}") + ("" if e.correct else "(x)")
            for e in events
        )
        _text(s, f"events: {log}", (x, y), 19, MUTED)


P_SCENE = pygame.Rect(0, 0, 560, 315)
P_KEYBOARD = pygame.Rect(0, 315, 560, 280)
P_STATUS = pygame.Rect(0, 595, 560, 125)
P_SCREEN = pygame.Rect(560, 0, 720, 450)
P_MOUSE = pygame.Rect(560, 450, 270, 270)
P_LEGEND = pygame.Rect(830, 450, 450, 270)
PRESS_HIGHLIGHT_S = 0.4  # keep a press highlighted this long (sim seconds) so frames catch it


class PerceptionView:
    """Close-up camera images with ground-truth labels: what a perception model learns from."""

    def __init__(self, env: ComputerUseEnv, main_camera: str = "overview"):
        self.env = env
        self.main_camera = main_camera
        self.surface = pygame.Surface((WIDTH, HEIGHT))

    def _recent_press(self) -> dict[str, str]:
        """{annotation name: caption} for the key / mouse button pressed in the last moment."""
        env = self.env
        recent = [e for e in env.tasks.events if env.data.time - e.time < PRESS_HIGHLIGHT_S]
        if not recent:
            return {}
        e = recent[-1]
        if e.kind == "key":
            key = KEY_BY_CHAR[e.value]
            return {key.body: f"pressed {key.label or key.name}"}
        _, button = split_click(e.value or "")
        return {f"mouse_{button}": f"{button} click"}

    def _camera_panel(self, s, camera: str, rect: pygame.Rect, title: str, highlight: dict, tags=()) -> list[dict]:
        image = self.env.render(camera, rect.height, rect.width)
        records = self.env.annotations(camera, rect.height, rect.width)
        _blit_array(s, draw_annotations(image, records, tags=tags, highlight=highlight), rect)
        _label(s, title, pygame.Rect(rect.x, rect.bottom - 34, rect.width, 30), 18)
        return records

    def draw(self, skill: str | None = None, policy_name: str = "") -> pygame.Surface:
        env, s = self.env, self.surface
        s.fill(BG)
        highlight = self._recent_press()
        _blit_array(s, env.render(self.main_camera, P_SCENE.height, P_SCENE.width), P_SCENE)
        _label(s, "3D scene", P_SCENE, 18)
        seen = self._camera_panel(s, "keyboard_cam", P_KEYBOARD, "keyboard_cam: 44 labelled keys", highlight)
        seen += self._camera_panel(s, "screen_cam", P_SCREEN, "screen_cam: every string on screen is labelled",
                                   highlight)
        seen += self._camera_panel(s, "mouse_cam", P_MOUSE, "mouse_cam: L / R", highlight, tags=("mouse_button",))
        self._draw_status(s)
        self._draw_legend(s, seen)
        for r in (P_KEYBOARD, P_SCREEN, P_MOUSE, P_LEGEND, P_STATUS):
            pygame.draw.rect(s, (40, 44, 54), r, 1)
        return s

    def frame(self, skill: str | None = None, policy_name: str = "") -> np.ndarray:
        return pygame.surfarray.array3d(self.draw(skill, policy_name)).transpose(1, 0, 2)

    def _draw_status(self, s) -> None:
        env = self.env
        x, y = P_STATUS.x + 14, P_STATUS.y + 10
        _text(s, env.task.instruction, (x, y), 24)
        y += 30
        if env.status.success:
            badge, color = "SUCCESS", GREEN
        elif env.status.timeout:
            badge, color = "TIMEOUT", RED
        else:
            badge, color = "IN PROGRESS", AMBER
        pygame.draw.rect(s, color, (x, y, 130, 26), border_radius=5)
        _text(s, badge, (x + 10, y + 6), 20, (15, 15, 15))
        _text(s, f"t = {env.elapsed:5.2f} s", (x + 145, y + 6), 19, MUTED)
        y += 34
        events = env.tasks.events[-7:]
        log = "  ".join(
            (repr(e.value)[1:-1] if e.kind == "key" else f"click:{e.value}") + ("" if e.correct else "(x)")
            for e in events
        )
        _text(s, f"events: {log}", (x, y), 19, MUTED)

    def _draw_legend(self, s, records: list[dict]) -> None:
        x, y = P_LEGEND.x + 16, P_LEGEND.y + 14
        _text(s, "Camera pixels + ground-truth labels", (x, y), 22)
        y += 26
        _text(s, "env.annotations(camera): label, box, visibility", (x, y), 17, MUTED)
        y += 30
        visible = [r for r in records if r["visible"] >= 0.5]
        count = lambda kind: sum(r["kind"] == kind for r in visible)
        rows = [
            (ANNOTATION_COLORS["key"], f"key legend    {count('key'):2d} visible"),
            (ANNOTATION_COLORS["mouse_button"], f"mouse button  {count('mouse_button'):2d} visible (L, R)"),
            (ANNOTATION_COLORS["screen_text"], f"screen text   {count('screen_text'):2d} strings"),
            (ANNOTATION_COLORS["cursor"], "cursor"),
            (HIGHLIGHT_COLOR, "key / button being pressed"),
        ]
        for color, text in rows:
            pygame.draw.rect(s, color, (x, y + 2, 16, 16), 3)
            _text(s, text, (x + 26, y), 19)
            y += 26
        hidden = sum(r["kind"] == "key" and r["visible"] < 0.5 for r in records)
        if hidden:
            _text(s, f"{hidden} keys hidden under the hand (visibility < 0.5)", (x, y + 4), 17, MUTED)
