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


def test_context_menus():
    d = Desktop()
    d.reset(active="browser")
    d.set_cursor((300, 250))
    assert d.click("right").button == "right"
    assert d.state()["menu"] == ["menu:back", "menu:home", "menu:close"]
    d.set_cursor((150, 100))
    assert d.click().target is None and d.menu is None  # clicking outside only dismisses it
    d.set_cursor((300, 250))
    d.click("right")
    _click(d, "menu:close")
    assert d.active is None
    d.set_cursor((300, 250))
    d.click("right")  # on the empty desktop the menu opens apps
    assert d.state()["menu"] == ["menu:open:browser", "menu:open:editor", "menu:open:files"]
    _click(d, "menu:open:files")
    assert d.active == "files"


def test_clear_text_task_uses_right_click_menu():
    d = Desktop()
    task = make_task("clear_text", "meeting notes")
    task.setup(d)
    assert d.apps["editor"].text == "meeting notes" and not task.success(d)
    assert task.expectation(d) == (None, "right:editor:text")
    d.set_cursor(d.widget_rect("editor:text").center)
    d.click("right")
    assert task.expectation(d) == (None, "menu:clear") and task.progress(d) == 0.5
    _click(d, "menu:clear")
    assert task.success(d) and task.progress(d) == 1.0


def test_screen_text_is_recorded_with_boxes():
    d = Desktop()
    d.reset(active="editor")
    for ch in "hello":
        d.key_press(ch)
    d.render()
    texts = {t["text"]: t["bbox"] for t in d.text_items}
    assert "hello" in texts and "Text Editor" in texts and "untitled.txt" in texts
    x0, y0, x1, y1 = texts["hello"]
    assert d.window_rect.contains((x0, y0, x1 - x0, y1 - y0))


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
    for name in ("type_text", "search", "open_result", "clear_text"):
        assert not set(SPLITS[name]["train"]) & set(SPLITS[name]["test"]), name
