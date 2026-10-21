"""Can a model read the scene? Legends, mouse buttons and screen text are recovered from the
rendered camera images alone (template matching on the annotated boxes), so they are
legible at the default camera sizes, and the annotations point at the right pixels."""

import numpy as np
import pygame
import pytest
from PIL import Image

from environment.keyboard_layout import KEYS
from environment.screen import MOUSE_BUTTON_RGB, MOUSE_LABELS, MOUSE_LEGEND_RGB, label_image
from environment.task_manager import make_task
from learning.scripted_policy import ScriptedPolicy


def _array(surface: pygame.Surface) -> np.ndarray:
    return pygame.surfarray.array3d(surface).transpose(1, 0, 2)


def _features(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Grey, resized to size=(W, H), zero-mean, unit-norm: dot product = correlation."""
    grey = np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).convert("L").resize(size, Image.BILINEAR), float)
    grey -= grey.mean()
    return grey.ravel() / (np.linalg.norm(grey) + 1e-9)


def _crop(image: np.ndarray, bbox, inset: float = 0.06) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    dx, dy = round((x1 - x0) * inset), round((y1 - y0) * inset)
    return image[y0 + dy : y1 - dy, x0 + dx : x1 - dx]


def _read(crop: np.ndarray, templates: dict[str, np.ndarray], size) -> str:
    feats = _features(crop, size)
    return max(templates, key=lambda label: float(feats @ templates[label]))


def _templates(cells: dict[str, np.ndarray], size) -> dict[str, np.ndarray]:
    return {label: _features(_crop(img, (0, 0, img.shape[1], img.shape[0])), size) for label, img in cells.items()}


def test_every_key_legend_is_readable_from_keyboard_cam(env):
    env.reset(make_task("type_key", "a"))  # the arm's home pose is beside the keyboard
    image = env.render("keyboard_cam", *env.camera_size("keyboard_cam"))
    records = {r["name"]: r for r in env.annotations("keyboard_cam") if r["kind"] == "key"}
    assert len(records) == len(KEYS) and all(r["visible"] > 0.9 for r in records.values())

    misread = []
    for width in {k.width_units for k in KEYS}:
        group = [k for k in KEYS if k.width_units == width]
        x0, y0, x1, y1 = group[0].atlas_rect()
        size = (round(24 * (x1 - x0) / (y1 - y0)), 24)
        cells = {k.label: _array(label_image(k.label, (x1 - x0, y1 - y0), k.cap_rgb, k.legend_rgb,
                                             0.72 if len(k.label) == 1 else 0.42)) for k in group}
        templates = _templates(cells, size)
        for k in group:
            read = _read(_crop(image, records[k.body]["bbox"]), templates, size)
            if read != k.label:
                misread.append((k.label, read))
    assert misread == []


def test_left_and_right_mouse_buttons_are_readable_from_mouse_cam(env):
    env.reset(make_task("type_key", "a"))
    image = env.render("mouse_cam", *env.camera_size("mouse_cam"))
    buttons = {r["button"]: r for r in env.annotations("mouse_cam") if r["kind"] == "mouse_button"}
    assert set(buttons) == {"left", "right"} and all(b["visible"] > 0.9 for b in buttons.values())
    size = (20, 22)
    templates = _templates({t: _array(label_image(t, (64, 72), MOUSE_BUTTON_RGB, MOUSE_LEGEND_RGB, 0.6))
                            for t in MOUSE_LABELS}, size)
    assert _read(_crop(image, buttons["left"]["bbox"]), templates, size) == "L"
    assert _read(_crop(image, buttons["right"]["bbox"]), templates, size) == "R"
    assert buttons["left"]["bbox"][0] < buttons["right"]["bbox"][0]  # L appears on the image's left


@pytest.mark.parametrize("task, text", [("search", "robotics research papers"), ("open_result", "mouse control")])
def test_screen_text_in_screen_cam_matches_the_display(env, task, text):
    env.reset(make_task(task, text))
    policy = ScriptedPolicy(env)
    for _ in range(350):  # get some typed text / results on screen
        env.step(policy.act())
    image = env.render("screen_cam", *env.camera_size("screen_cam"))
    display = env.screen_image()
    scores = []
    for r in env.annotations("screen_cam"):
        sx0, sy0, sx1, sy1 = r.get("screen_bbox", (0, 0, 0, 0))
        if r["kind"] != "screen_text" or r["visible"] < 0.95 or sy1 - sy0 < 12 or sx1 - sx0 < 12:
            continue
        size = (sx1 - sx0, sy1 - sy0)
        truth = _features(display[sy0:sy1, sx0:sx1], size)
        scores.append(float(_features(_crop(image, r["bbox"], 0.0), size) @ truth))
    assert len(scores) >= 5
    assert min(scores) > 0.95, scores  # screen_cam shows the display 1:1, boxes pixel-exact


def test_annotations_report_keys_hidden_by_the_hand(env):
    env.reset(make_task("type_key", "g"))
    policy = ScriptedPolicy(env)
    while env.controller.finger_extension(env.data) < 0.005:  # hand hovering over "g"
        env.step(policy.act())
    visible = {r["label"]: r["visible"] for r in env.annotations("keyboard_cam") if r["kind"] == "key"}
    assert visible["G"] < 0.5  # under the finger
    assert visible["1"] > 0.9 and visible["P"] > 0.9  # far from the hand
