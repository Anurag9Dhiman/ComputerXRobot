"""Simulated desktop logic, task definitions and the keyboard asset."""

from environment.browser import Desktop
from environment.keyboard_layout import DEFAULT_PATH, KEYS, TYPEABLE_CHARS, generate_keyboard_xml
from environment.task_manager import SPLITS, make_task


def _click(desktop, widget):
    desktop.set_cursor(desktop.widget_rect(widget).center)
    return desktop.click()


def test_browser_search_flow():
    d = Desktop()
    task = make_task("search", "robotics research papers")
    task.setup(d)
    assert task.expectation(d) == (None, "browser:search_field")
    assert _click(d, "browser:search_field").target == "browser:search_field"
    for ch in "robotics":
        assert task.expectation(d) == (ch, None)
        d.key_press(ch)
    d.key_press("x")  # typo -> task asks for backspace
    assert task.expectation(d) == ("\b", None)
    d.key_press("\b")
    for ch in " research papers":
        d.key_press(ch)
    assert task.expectation(d) == ("\n", None)
    d.key_press("\n")
    assert task.success(d)
    assert d.apps["browser"].results[0].id == "robot-typing"


def test_open_result_and_back():
    d = Desktop()
    task = make_task("open_result", "mouse control")
    task.setup(d)
    _click(d, "browser:search_field")
    for ch in "mouse control\n":
        d.key_press(ch)
    _, click = task.expectation(d)
    _click(d, click)
    assert task.success(d)
    _click(d, "browser:back")
    assert d.apps["browser"].page == "results"


def test_dock_and_files_open_apps():
    d = Desktop()
    d.reset(active=None)
    _click(d, "dock:files")
    assert d.active == "files"
    _click(d, "files:item:1")
    assert d.active == "editor" and d.apps["editor"].filename == "todo.txt"
    _click(d, "window:close")
    assert d.active is None


def test_keys_typed_without_focus_are_rejected():
    d = Desktop()
    d.reset(active="browser")
    assert not d.key_press("a").accepted
    assert d.apps["browser"].query == ""


def test_keyboard_xml_matches_layout():
    assert DEFAULT_PATH.read_text() == generate_keyboard_xml(), "run: python -m environment.keyboard_layout"
    assert len({k.char for k in KEYS}) == len(KEYS)


def test_training_text_covers_every_key():
    seen = set("".join(SPLITS["type_text"]["train"]))
    assert set(TYPEABLE_CHARS) <= seen


def test_test_split_is_unseen():
    for name in ("type_text", "search", "open_result"):
        assert not set(SPLITS[name]["train"]) & set(SPLITS[name]["test"]), name
