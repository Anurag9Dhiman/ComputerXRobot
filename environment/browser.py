"""Simulated desktop shown on the monitor: a dock, a text editor, a file manager and a
web browser with an offline search engine.

The desktop is a plain state machine driven by three inputs -- `move_cursor`, `click` and
`key_press` -- and rendered with pygame into a 640x400 surface, which environment/screen.py
copies into the monitor texture. Nothing here knows about physics; environment/ui_events.py
translates physical key/mouse interaction into these calls.

Every clickable element has a widget id (e.g. "browser:search_field", "dock:editor") and a
screen rectangle, so tasks can say what should be clicked and the scripted expert can find it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from environment.web_content import WebPage, search

SCREEN_W, SCREEN_H = 640, 400
DOCK_W = 60
TITLE_H = 24
DOCK_APPS = (("browser", "Browser"), ("editor", "Editor"), ("files", "Files"))

C_DESKTOP = (44, 82, 138)
C_DOCK = (24, 30, 42)
C_WINDOW = (250, 250, 252)
C_TITLE = (226, 229, 235)
C_TEXT = (30, 32, 36)
C_MUTED = (110, 116, 126)
C_BORDER = (190, 195, 205)
C_FOCUS = (40, 110, 230)
C_LINK = (26, 72, 190)
C_URL = (20, 120, 60)
C_BUTTON = (40, 110, 230)


class Fonts:
    """Lazily created pygame fonts (pygame's bundled default font, so no system fonts needed)."""

    _cache: dict[int, pygame.font.Font] = {}

    @classmethod
    def get(cls, size: int) -> pygame.font.Font:
        if size not in cls._cache:
            if not pygame.font.get_init():
                pygame.font.init()
            cls._cache[size] = pygame.font.Font(None, size)
        return cls._cache[size]


def _text(surf, text, pos, size=18, color=C_TEXT, center=False):
    img = Fonts.get(size).render(text, True, color)
    rect = img.get_rect(center=pos) if center else img.get_rect(topleft=pos)
    surf.blit(img, rect)
    return rect


def _wrap(text: str, size: int, width: int) -> list[str]:
    font = Fonts.get(size)
    lines: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split(" "):
            candidate = word if not line else f"{line} {word}"
            if font.size(candidate)[0] <= width or not line:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


@dataclass
class ClickResult:
    target: str | None  # widget id under the cursor, None if the click hit nothing
    pos: tuple[int, int]


@dataclass
class KeyResult:
    char: str
    accepted: bool  # False if no focused text field received the key
    target: str | None


# ---------------------------------------------------------------------------------- apps
class App:
    name = "app"
    title = "App"

    def __init__(self, desktop: "Desktop"):
        self.desktop = desktop
        self.reset()

    def reset(self) -> None:
        pass

    def layout(self, content: pygame.Rect) -> dict[str, pygame.Rect]:
        return {}

    def draw(self, surf: pygame.Surface, content: pygame.Rect) -> None:
        pass

    def click(self, widget: str | None) -> None:
        pass

    def key(self, char: str) -> bool:
        return False

    @property
    def focused_widget(self) -> str | None:
        return None


class TextEditor(App):
    name = "editor"
    title = "Text Editor"

    def reset(self) -> None:
        self.filename = "untitled.txt"
        self.text = ""
        self.focused = True

    @property
    def focused_widget(self) -> str | None:
        return "editor:text" if self.focused else None

    def open_file(self, filename: str, text: str) -> None:
        self.filename, self.text, self.focused = filename, text, True

    def layout(self, content):
        return {"editor:text": content.inflate(-12, -12)}

    def click(self, widget):
        self.focused = widget == "editor:text"

    def key(self, char):
        if not self.focused:
            return False
        if char == "\b":
            self.text = self.text[:-1]
        else:
            self.text += char
        return True

    def draw(self, surf, content):
        area = self.layout(content)["editor:text"]
        pygame.draw.rect(surf, (255, 255, 255), area)
        pygame.draw.rect(surf, C_FOCUS if self.focused else C_BORDER, area, 2 if self.focused else 1)
        _text(surf, self.filename, (area.x + 6, area.y + 6), 16, C_MUTED)
        y = area.y + 28
        lines = _wrap(self.text, 30, area.width - 16) if self.text else [""]
        for line in lines[-9:]:
            _text(surf, line, (area.x + 8, y), 30)
            y += 26
        if self.focused:
            last = lines[-1]
            x = area.x + 8 + Fonts.get(30).size(last)[0] + 1
            pygame.draw.line(surf, C_TEXT, (x, y - 26), (x, y - 6), 2)


class FileManager(App):
    name = "files"
    title = "Files"
    FILES = (
        ("notes.txt", "meeting at 10\nrobot demo on friday"),
        ("todo.txt", "collect demos\ntrain policy\nevaluate"),
        ("papers.txt", "robot typing\ncomputer use in simulation"),
    )

    def layout(self, content):
        return {
            f"files:item:{i}": pygame.Rect(content.x + 16, content.y + 44 + i * 40, 300, 32)
            for i in range(len(self.FILES))
        }

    def click(self, widget):
        if widget and widget.startswith("files:item:"):
            name, text = self.FILES[int(widget.rsplit(":", 1)[1])]
            editor = self.desktop.apps["editor"]
            editor.open_file(name, text)
            self.desktop.open_app("editor")

    def draw(self, surf, content):
        _text(surf, "Home / Documents", (content.x + 16, content.y + 14), 18, C_MUTED)
        for (name, _), rect in zip(self.FILES, self.layout(content).values()):
            pygame.draw.rect(surf, (238, 241, 246), rect, border_radius=4)
            pygame.draw.rect(surf, (120, 140, 170), (rect.x + 8, rect.y + 7, 14, 18), border_radius=2)
            _text(surf, name, (rect.x + 32, rect.y + 9), 20)


class Browser(App):
    name = "browser"
    title = "Browser"
    MAX_QUERY = 48

    def reset(self) -> None:
        self.page = "home"  # "home" | "results" | "doc"
        self.query = ""  # text in the search field
        self.field_focused = False
        self.last_query: str | None = None  # last submitted query
        self.results: list[WebPage] = []
        self.doc: WebPage | None = None
        self.history: list[tuple] = []

    @property
    def focused_widget(self) -> str | None:
        return "browser:search_field" if self.field_focused else None

    @property
    def url(self) -> str:
        if self.page == "results":
            return f"seeker://search?q={self.last_query.replace(' ', '+')}"
        if self.page == "doc" and self.doc:
            return f"https://{self.doc.url}"
        return "seeker://home"

    def _snapshot(self):
        return (self.page, self.last_query, list(self.results), self.doc)

    def layout(self, content):
        w = {"browser:back": pygame.Rect(content.x + 6, content.y + 5, 24, 22)}
        if self.page == "home":
            cx = content.centerx
            w["browser:search_field"] = pygame.Rect(cx - 190, content.y + 160, 300, 32)
            w["browser:search_button"] = pygame.Rect(cx + 118, content.y + 160, 72, 32)
        elif self.page == "results":
            w["browser:search_field"] = pygame.Rect(content.x + 16, content.y + 40, 300, 26)
            w["browser:search_button"] = pygame.Rect(content.x + 322, content.y + 40, 64, 26)
            for i, _ in enumerate(self.results):
                w[f"browser:result:{i}"] = pygame.Rect(content.x + 16, content.y + 96 + i * 52, 440, 18)
        return w

    def submit(self) -> None:
        query = " ".join(self.query.split())
        if not query:
            return
        self.history.append(self._snapshot())
        self.last_query = query
        self.results = search(query)
        self.page, self.doc, self.field_focused = "results", None, False

    def open_result(self, index: int) -> None:
        self.history.append(self._snapshot())
        self.doc = self.results[index]
        self.page, self.field_focused = "doc", False

    def back(self) -> None:
        if self.history:
            self.page, self.last_query, self.results, self.doc = self.history.pop()
            self.query = self.last_query or ""
        self.field_focused = False

    def click(self, widget):
        if widget == "browser:search_field":
            self.field_focused = True
            return
        self.field_focused = False
        if widget == "browser:search_button":
            self.submit()
        elif widget == "browser:back":
            self.back()
        elif widget and widget.startswith("browser:result:"):
            self.open_result(int(widget.rsplit(":", 1)[1]))

    def key(self, char):
        if not self.field_focused:
            return False
        if char == "\b":
            self.query = self.query[:-1]
        elif char == "\n":
            self.submit()
        elif len(self.query) < self.MAX_QUERY:
            self.query += char
        return True

    # ---- drawing
    def draw(self, surf, content):
        w = self.layout(content)
        bar = pygame.Rect(content.x, content.y, content.width, 32)
        pygame.draw.rect(surf, (236, 238, 242), bar)
        back = w["browser:back"]
        pygame.draw.rect(surf, (214, 218, 226), back, border_radius=4)
        _text(surf, "<", back.center, 22, C_TEXT, center=True)
        addr = pygame.Rect(back.right + 8, content.y + 5, content.width - 48, 22)
        pygame.draw.rect(surf, (255, 255, 255), addr, border_radius=11)
        _text(surf, self.url, (addr.x + 10, addr.y + 5), 16, C_MUTED)

        if self.page == "home":
            self._draw_logo(surf, (content.centerx, content.y + 110))
            self._draw_field(surf, w["browser:search_field"], w["browser:search_button"], 22)
        elif self.page == "results":
            self._draw_field(surf, w["browser:search_field"], w["browser:search_button"], 18)
            n = len(self.results)
            _text(surf, f"{n} result{'s' if n != 1 else ''}" if n else "No results found",
                  (content.x + 16, content.y + 74), 15, C_MUTED)
            for i, page in enumerate(self.results):
                r = w[f"browser:result:{i}"]
                _text(surf, page.title, r.topleft, 20, C_LINK)
                _text(surf, page.url, (r.x, r.y + 17), 14, C_URL)
                _text(surf, page.snippet[:78], (r.x, r.y + 30), 14, C_MUTED)
        elif self.page == "doc" and self.doc:
            _text(surf, self.doc.title, (content.x + 16, content.y + 44), 26)
            _text(surf, self.doc.url, (content.x + 16, content.y + 68), 15, C_URL)
            y = content.y + 94
            for line in _wrap(self.doc.snippet + "\n\n" + self.doc.body, 19, content.width - 32):
                _text(surf, line, (content.x + 16, y), 19)
                y += 18

    def _draw_logo(self, surf, center):
        colors = [(66, 133, 244), (219, 68, 55), (244, 180, 0), (66, 133, 244), (15, 157, 88), (219, 68, 55)]
        font = Fonts.get(56)
        word = "Seeker"
        width = font.size(word)[0]
        x = center[0] - width // 2
        for ch, color in zip(word, colors):
            img = font.render(ch, True, color)
            surf.blit(img, img.get_rect(midleft=(x, center[1])))
            x += font.size(ch)[0]

    def _draw_field(self, surf, field_rect, button_rect, size):
        pygame.draw.rect(surf, (255, 255, 255), field_rect, border_radius=6)
        pygame.draw.rect(surf, C_FOCUS if self.field_focused else C_BORDER, field_rect,
                         2 if self.field_focused else 1, border_radius=6)
        if self.query or self.field_focused:
            _text(surf, self.query, (field_rect.x + 10, field_rect.y + (field_rect.height - size * 0.7) / 2),
                      size)
            if self.field_focused:
                x = field_rect.x + 10 + Fonts.get(size).size(self.query)[0] + 1
                pygame.draw.line(surf, C_TEXT, (x, field_rect.y + 6), (x, field_rect.bottom - 6), 2)
        else:
            _text(surf, "Search the web", (field_rect.x + 10, field_rect.y + (field_rect.height - size * 0.7) / 2),
                  size, C_MUTED)
        pygame.draw.rect(surf, C_BUTTON, button_rect, border_radius=6)
        _text(surf, "Search", button_rect.center, size, (255, 255, 255), center=True)


# ------------------------------------------------------------------------------- desktop
class Desktop:
    def __init__(self, size: tuple[int, int] = (SCREEN_W, SCREEN_H)):
        self.size = size
        self.apps: dict[str, App] = {cls.name: cls(self) for cls in (Browser, TextEditor, FileManager)}
        self._surface = pygame.Surface(self.size)
        self.reset()

    def reset(self, active: str | None = None) -> None:
        for app in self.apps.values():
            app.reset()
        self.active = active
        self.cursor = (self.size[0] // 2, self.size[1] // 2)
        self._cursor_f = [float(self.cursor[0]), float(self.cursor[1])]
        self.version = 0  # bumps on every visible change
        self._rendered_version = -1

    def _changed(self) -> None:
        self.version += 1

    # ---- window management
    def open_app(self, name: str) -> None:
        self.active = name
        self._changed()

    def close_app(self) -> None:
        self.active = None
        self._changed()

    @property
    def app(self) -> App | None:
        return self.apps.get(self.active) if self.active else None

    @property
    def window_rect(self) -> pygame.Rect:
        return pygame.Rect(DOCK_W + 8, 8, self.size[0] - DOCK_W - 16, self.size[1] - 16)

    @property
    def content_rect(self) -> pygame.Rect:
        w = self.window_rect
        return pygame.Rect(w.x, w.y + TITLE_H, w.width, w.height - TITLE_H)

    def widgets(self) -> dict[str, pygame.Rect]:
        """All clickable widgets in hit-test order (topmost first)."""
        out: dict[str, pygame.Rect] = {}
        if self.app:
            out.update(self.app.layout(self.content_rect))
            w = self.window_rect
            out["window:close"] = pygame.Rect(w.right - 26, w.y + 3, 22, 18)
        for i, (name, _) in enumerate(DOCK_APPS):
            out[f"dock:{name}"] = pygame.Rect(10, 14 + i * 66, 40, 40)
        return out

    def widget_rect(self, name: str) -> pygame.Rect | None:
        return self.widgets().get(name)

    def widget_at(self, pos) -> str | None:
        for name, rect in self.widgets().items():
            if rect.collidepoint(pos):
                return name
        return None

    @property
    def focused_widget(self) -> str | None:
        return self.app.focused_widget if self.app else None

    # ---- input
    def move_cursor(self, du: float, dv: float) -> None:
        self._cursor_f[0] = min(max(self._cursor_f[0] + du, 0.0), self.size[0] - 1.0)
        self._cursor_f[1] = min(max(self._cursor_f[1] + dv, 0.0), self.size[1] - 1.0)
        new = (int(round(self._cursor_f[0])), int(round(self._cursor_f[1])))
        if new != self.cursor:
            self.cursor = new
            self._changed()

    def set_cursor(self, pos) -> None:
        self._cursor_f = [float(pos[0]), float(pos[1])]
        self.move_cursor(0, 0)

    def click(self) -> ClickResult:
        target = self.widget_at(self.cursor)
        if target and target.startswith("dock:"):
            self.open_app(target.split(":", 1)[1])
        elif target == "window:close":
            self.close_app()
        elif self.app and self.window_rect.collidepoint(self.cursor):
            self.app.click(target)
        self._changed()
        return ClickResult(target, self.cursor)

    def key_press(self, char: str) -> KeyResult:
        accepted = bool(self.app and self.app.key(char))
        if accepted:
            self._changed()
        return KeyResult(char, accepted, self.focused_widget)

    # ---- rendering
    def render(self) -> pygame.Surface:
        if self._rendered_version == self.version:
            return self._surface
        s = self._surface
        s.fill(C_DESKTOP)
        pygame.draw.rect(s, C_DOCK, (0, 0, DOCK_W, self.size[1]))
        widgets = self.widgets()
        for name, label in DOCK_APPS:
            self._draw_icon(s, name, widgets[f"dock:{name}"], active=self.active == name)
            _text(s, label, (widgets[f"dock:{name}"].centerx, widgets[f"dock:{name}"].bottom + 9), 14,
                  (220, 225, 235), center=True)
        if self.app:
            w = self.window_rect
            pygame.draw.rect(s, (20, 30, 50), w.move(3, 3), border_radius=6)
            pygame.draw.rect(s, C_WINDOW, w, border_radius=6)
            pygame.draw.rect(s, C_TITLE, (w.x, w.y, w.width, TITLE_H), border_top_left_radius=6,
                             border_top_right_radius=6)
            _text(s, self.app.title, (w.x + 10, w.y + 6), 18, C_TEXT)
            close = widgets["window:close"]
            pygame.draw.rect(s, (225, 85, 75), close, border_radius=4)
            _text(s, "x", close.center, 18, (255, 255, 255), center=True)
            self.app.draw(s, self.content_rect)
        else:
            _text(s, "Click an icon to open an app", (self.size[0] // 2 + DOCK_W // 2, self.size[1] // 2), 22,
                  (200, 210, 230), center=True)
        self._draw_cursor(s)
        self._rendered_version = self.version
        return s

    @staticmethod
    def _draw_icon(s, name, rect, active):
        pygame.draw.rect(s, (70, 80, 100) if active else (48, 56, 72), rect.inflate(6, 6), border_radius=8)
        if name == "browser":
            pygame.draw.circle(s, (66, 133, 244), rect.center, 17)
            pygame.draw.ellipse(s, (15, 157, 88), (rect.x + 9, rect.y + 6, 14, 18))
            pygame.draw.ellipse(s, (15, 157, 88), (rect.x + 22, rect.y + 20, 12, 12))
        elif name == "editor":
            page = pygame.Rect(rect.x + 8, rect.y + 4, 24, 32)
            pygame.draw.rect(s, (245, 245, 245), page, border_radius=2)
            for i in range(4):
                pygame.draw.line(s, (120, 130, 145), (page.x + 4, page.y + 7 + i * 6), (page.right - 4, page.y + 7 + i * 6), 2)
        elif name == "files":
            pygame.draw.rect(s, (230, 175, 60), (rect.x + 4, rect.y + 8, 16, 8), border_radius=2)
            pygame.draw.rect(s, (245, 195, 80), (rect.x + 4, rect.y + 12, 32, 22), border_radius=3)

    def _draw_cursor(self, s):
        x, y = self.cursor
        pts = [(0, 0), (0, 17), (4, 13), (7, 20), (10, 19), (7, 12), (12, 12)]
        poly = [(x + px, y + py) for px, py in pts]
        pygame.draw.polygon(s, (255, 255, 255), poly)
        pygame.draw.polygon(s, (0, 0, 0), poly, 1)

    # ---- introspection
    def state(self) -> dict:
        browser, editor = self.apps["browser"], self.apps["editor"]
        return {
            "active_app": self.active,
            "cursor": self.cursor,
            "focused": self.focused_widget,
            "editor_text": editor.text,
            "editor_file": editor.filename,
            "browser_page": browser.page,
            "browser_query": browser.query,
            "browser_last_query": browser.last_query,
            "browser_doc": browser.doc.id if browser.doc else None,
        }


