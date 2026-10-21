"""Demo dashboard: the 3D scene, the robot's camera, the simulated screen and the task
result in one frame. Used for the live window (`--display`) and for videos (`--video`).

    +------------------------------+--------------------+
    |                              |  robot camera      |
    |        3D scene (overview)   +--------------------+
    |                              |  simulated screen  |
    +------------------------------+----------+----+----+
    |  task / status / metrics     | keyboard |mouse|wrist|
    +------------------------------+----------+----+----+

The keyboard and mouse panels are the legible close-up cameras (keyboard_cam, mouse_cam).
"""

from __future__ import annotations

import numpy as np
import pygame

from environment.browser import Fonts
from environment.computer_env import ComputerUseEnv

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
