"""Tasks, success checks, rewards and episode bookkeeping.

A task puts the desktop into its initial state, says what it wants next
(`expectation()` -> (char to type, click target); a click target "right:<widget>" means a
right-click) and decides success from the desktop
state alone -- never from what the robot did -- so scripted and learned policies are judged
the same way.

Task kinds:
    type_key     press one key; the character must appear in the text editor (milestone 1)
    type_text    type a string into the text editor
    search       click the search field, type a query, press Enter, see results
    open_result  search, then open a specific result page
    open_app     open an app by clicking its dock icon
    clear_text   clear prefilled editor text with the right-click menu (right-click, then
                 left-click "Clear all")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from environment.browser import Desktop, click_target, split_click
from environment.keyboard_layout import TYPEABLE_CHARS
from environment.ui_events import UIEvent
from environment.web_content import PAGE_BY_ID, search


def _typing_expectation(current: str, target: str) -> str | None:
    """Next key to press to turn `current` into `target` (backspace on a typo)."""
    if not target.startswith(current):
        return "\b"
    return target[len(current)] if len(current) < len(target) else None


def _prefix_fraction(current: str, target: str) -> float:
    n = 0
    for a, b in zip(current, target):
        if a != b:
            break
        n += 1
    return n / max(len(target), 1)


class Task:
    name = "task"

    @property
    def instruction(self) -> str:
        raise NotImplementedError

    @property
    def time_limit(self) -> float:
        return 20.0

    def setup(self, desktop: Desktop) -> None:
        raise NotImplementedError

    def expectation(self, desktop: Desktop) -> tuple[str | None, str | None]:
        """(character to type next, widget to click next); both None when nothing is needed."""
        raise NotImplementedError

    def progress(self, desktop: Desktop) -> float:
        return float(self.success(desktop))

    def success(self, desktop: Desktop) -> bool:
        raise NotImplementedError

    def spec(self) -> dict:
        return {"task": self.name}


class TypeTextTask(Task):
    name = "type_text"

    def __init__(self, text: str):
        bad = set(text) - set(TYPEABLE_CHARS)
        if bad:
            raise ValueError(f"cannot type {sorted(bad)} on the simulated keyboard")
        self.text = text

    @property
    def instruction(self):
        return f'Type "{self.text}" in the text editor'

    @property
    def time_limit(self):
        return 6.0 + 1.0 * len(self.text)

    def setup(self, desktop):
        desktop.reset(active="editor")

    def expectation(self, desktop):
        editor = desktop.apps["editor"]
        if desktop.active != "editor":
            return None, "dock:editor"
        if not editor.focused:
            return None, "editor:text"
        return _typing_expectation(editor.text, self.text), None

    def progress(self, desktop):
        return _prefix_fraction(desktop.apps["editor"].text, self.text)

    def success(self, desktop):
        return desktop.apps["editor"].text == self.text

    def spec(self):
        return {"task": self.name, "text": self.text}


class TypeKeyTask(TypeTextTask):
    """Milestone 1: one physical key press -> one character in the editor."""

    name = "type_key"

    def __init__(self, text: str):
        if len(text) != 1:
            raise ValueError("type_key expects exactly one character")
        super().__init__(text)

    @property
    def instruction(self):
        return f'Press the "{self.text}" key'


class SearchTask(Task):
    name = "search"

    def __init__(self, text: str):
        bad = set(text) - set(TYPEABLE_CHARS)
        if bad:
            raise ValueError(f"cannot type {sorted(bad)} on the simulated keyboard")
        self.text = text

    @property
    def instruction(self):
        return f'Search the web for "{self.text}"'

    @property
    def time_limit(self):
        return 15.0 + 1.0 * len(self.text)

    def setup(self, desktop):
        desktop.reset(active="browser")

    def _searched(self, desktop) -> bool:
        browser = desktop.apps["browser"]
        return browser.page == "results" and browser.last_query == self.text

    def expectation(self, desktop):
        browser = desktop.apps["browser"]
        if self._searched(desktop):
            return None, None
        if desktop.active != "browser":
            return None, "dock:browser"
        if browser.page == "doc":
            return None, "browser:back"
        if not browser.field_focused:
            return None, "browser:search_field"
        return _typing_expectation(browser.query, self.text) or "\n", None

    def progress(self, desktop):
        browser = desktop.apps["browser"]
        if self._searched(desktop):
            return 1.0
        return 0.2 * browser.field_focused + 0.7 * _prefix_fraction(browser.query, self.text)

    def success(self, desktop):
        return self._searched(desktop)

    def spec(self):
        return {"task": self.name, "text": self.text}


class OpenResultTask(SearchTask):
    """Search for `text` and open the result page `page_id` (default: the top hit)."""

    name = "open_result"

    def __init__(self, text: str, page_id: str | None = None):
        super().__init__(text)
        hits = search(text)
        if not hits:
            raise ValueError(f"query {text!r} has no results")
        self.page_id = page_id or hits[0].id
        if self.page_id not in [h.id for h in hits]:
            raise ValueError(f"{self.page_id!r} is not in the results for {text!r}")

    @property
    def instruction(self):
        return f'Search for "{self.text}" and open "{PAGE_BY_ID[self.page_id].title}"'

    @property
    def time_limit(self):
        return 25.0 + 1.0 * len(self.text)

    def expectation(self, desktop):
        browser = desktop.apps["browser"]
        if self.success(desktop):
            return None, None
        if self._searched(desktop):
            index = [r.id for r in browser.results].index(self.page_id)
            return None, f"browser:result:{index}"
        return super().expectation(desktop)

    def progress(self, desktop):
        return 1.0 if self.success(desktop) else 0.8 * super().progress(desktop)

    def success(self, desktop):
        browser = desktop.apps["browser"]
        return browser.page == "doc" and browser.doc is not None and browser.doc.id == self.page_id

    def spec(self):
        return {"task": self.name, "text": self.text, "page_id": self.page_id}


class OpenAppTask(Task):
    name = "open_app"

    def __init__(self, text: str):
        if text not in ("browser", "editor", "files"):
            raise ValueError(f"unknown app {text!r}")
        self.text = text

    @property
    def instruction(self):
        return f"Open the {self.text} app"

    @property
    def time_limit(self):
        return 12.0

    def setup(self, desktop):
        desktop.reset(active=None)

    def expectation(self, desktop):
        return (None, None) if self.success(desktop) else (None, f"dock:{self.text}")

    def success(self, desktop):
        return desktop.active == self.text

    def spec(self):
        return {"task": self.name, "text": self.text}


class ClearTextTask(Task):
    """The editor starts with `text` in it; empty it using the right-click context menu."""

    name = "clear_text"

    def __init__(self, text: str):
        if not text:
            raise ValueError("clear_text needs non-empty starting text")
        self.text = text

    @property
    def instruction(self):
        return "Clear the editor with the right-click menu"

    @property
    def time_limit(self):
        return 15.0

    def setup(self, desktop):
        desktop.reset(active="editor")
        desktop.apps["editor"].text = self.text

    def expectation(self, desktop):
        if self.success(desktop):
            return None, None
        if desktop.active != "editor":
            return None, "dock:editor"
        if desktop.menu is not None and "menu:clear" in dict(desktop.menu.items):
            return None, "menu:clear"
        return None, click_target("editor:text", "right")

    def progress(self, desktop):
        if self.success(desktop):
            return 1.0
        return 0.5 if desktop.menu is not None and "menu:clear" in dict(desktop.menu.items) else 0.0

    def success(self, desktop):
        return desktop.active == "editor" and desktop.apps["editor"].text == ""

    def spec(self):
        return {"task": self.name, "text": self.text}


TASKS: dict[str, type[Task]] = {
    cls.name: cls for cls in (TypeKeyTask, TypeTextTask, SearchTask, OpenResultTask, OpenAppTask, ClearTextTask)
}

# Disjoint train/test content, so evaluation on the "test" split measures generalisation to
# strings the policy never saw during training (every key appears in the training words).
SPLITS: dict[str, dict[str, list[str]]] = {
    "type_key": {
        "train": list("abcdefghijklmnopqrstuvwxyz0123456789"),
        "test": list("abcdefghijklmnopqrstuvwxyz0123456789"),
    },
    "type_text": {
        "train": ["quick", "brown", "fox", "jumps", "lazy", "dog", "robot", "mujoco", "pixel", "zebra",
                  "vortex", "hello world", "key 42", "a.b,c", "x-ray", "06/27", "13579", "8 hz;"],
        "test": ["arm", "mouse", "type this", "jazz", "sky", "bvqx", "2048", "web page", "click"],
    },
    "search": {
        "train": ["robotics", "mujoco tutorial", "pasta recipe", "weather today", "diffusion policy",
                  "python lists", "travel tokyo", "reinforcement learning"],
        "test": ["robotics research papers", "mouse control", "vision transformer", "imitation learning"],
    },
    "open_result": {
        "train": ["mujoco tutorial", "pasta recipe", "weather today"],
        "test": ["robotics research papers", "mouse control"],
    },
    "open_app": {"train": ["browser", "editor", "files"], "test": ["browser", "editor", "files"]},
    "clear_text": {
        "train": ["meeting notes", "draft 1", "hello there", "todo: buy milk", "x"],
        "test": ["shopping list", "quick note 42", "abc"],
    },
}


def make_task(name: str, text: str | None = None, **kwargs) -> Task:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; choose from {sorted(TASKS)}")
    if text is None:
        text = SPLITS[name]["test"][0]
    return TASKS[name](text, **kwargs)


def sample_task(name: str, split: str, rng: np.random.Generator) -> Task:
    options = SPLITS[name][split]
    return make_task(name, options[int(rng.integers(len(options)))])


# ---------------------------------------------------------------------------- manager
@dataclass
class TaskStatus:
    success: bool
    timeout: bool
    progress: float
    reward: float

    @property
    def done(self) -> bool:
        return self.success or self.timeout


@dataclass
class TaskManager:
    """Owns the current task: success checks, reward and event bookkeeping."""

    desktop: Desktop
    task: Task | None = None
    events: list[UIEvent] = field(default_factory=list)

    def reset(self, task: Task, time: float = 0.0) -> None:
        self.task = task
        task.setup(self.desktop)
        self.start_time = time
        self.events = []
        self.wrong_keys = 0
        self.failed_clicks = 0
        self.success_time: float | None = None
        self._last_progress = task.progress(self.desktop)
        self._new_errors = 0

    def expectation(self) -> tuple[str | None, str | None]:
        return self.task.expectation(self.desktop)

    def record(self, events: list[UIEvent]) -> None:
        for ev in events:
            self.events.append(ev)
            if not ev.correct:
                self._new_errors += 1
                if ev.kind == "key":
                    self.wrong_keys += 1
                else:
                    self.failed_clicks += 1

    def evaluate(self, time: float) -> TaskStatus:
        success = self.task.success(self.desktop)
        if success and self.success_time is None:
            self.success_time = time - self.start_time
        progress = self.task.progress(self.desktop)
        reward = (progress - self._last_progress) + (1.0 if success else 0.0) - 0.1 * self._new_errors
        self._last_progress, self._new_errors = progress, 0
        timeout = not success and time - self.start_time >= self.task.time_limit
        return TaskStatus(success, timeout, progress, reward)

    def features(self) -> dict:
        """Task-state observation: what to type / click next, with which mouse button, and
        where the target is on screen."""
        char, click = self.expectation()
        widget, button = split_click(click) if click else (None, None)
        rect = self.desktop.widget_rect(widget) if widget else None
        return {
            "expected_char": char,
            "click_target": click,
            "click_button": button,
            "click_target_px": rect.center if rect else None,
        }
