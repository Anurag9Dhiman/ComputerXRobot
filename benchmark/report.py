"""Tables from result files (Markdown or LaTeX).

Results of the same agent and track (e.g. several training seeds of `bc`) are grouped: a
single run shows score with its bootstrap 95% CI half-width, several runs show mean +- std
over runs. Scores are percentages.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark.suite import SPLITS, TASK_ORDER, TEST_SPLITS
from environment.task_manager import TASKS

TABLES = ("splits", "tasks", "levels", "errors")


def load(paths) -> list[dict]:
    return [json.loads(Path(p).read_text()) for p in paths]


def _label(r: dict) -> str:
    return r["agent_name"] + (" (privileged)" if r.get("privileged") else "")


def _group(results: list[dict]) -> dict[tuple[str, str], dict[str, list[dict]]]:
    groups: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        groups[(_label(r), r["track"])][r["split"]].append(r)
    # Privileged oracle last, then by name.
    return dict(sorted(groups.items(), key=lambda kv: ("privileged" in kv[0][0], kv[0])))


def _cell(values: list[float], cis: list[tuple[float, float]], fmt: str) -> str:
    pm = r"$\pm$" if fmt == "latex" else "±"
    if not values:
        return "–"
    if len(values) == 1:
        lo, hi = cis[0]
        return f"{100 * values[0]:.1f} {pm} {100 * (hi - lo) / 2:.1f}"
    return f"{100 * np.mean(values):.1f} {pm} {100 * np.std(values):.1f}"


def _render(header: list[str], rows: list[list[str]], fmt: str, caption: str) -> str:
    if fmt == "latex":
        cols = "l" * 2 + "c" * (len(header) - 2)
        esc = lambda s: s.replace("_", r"\_")
        lines = [r"\begin{table}[t]", r"\centering", f"\\caption{{{esc(caption)}}}", f"\\begin{{tabular}}{{{cols}}}",
                 r"\toprule", " & ".join(map(esc, header)) + r" \\", r"\midrule"]
        lines += [" & ".join(map(esc, row)) + r" \\" for row in rows]
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)
    lines = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def splits_table(results: list[dict], fmt: str = "md") -> str:
    splits = [s for s in (*TEST_SPLITS, "val", "train") if any(r["split"] == s for r in results)]
    rows = []
    for (label, track), by_split in _group(results).items():
        row = [label, track]
        for s in splits:
            rs = by_split.get(s, [])
            row.append(_cell([r["summary"]["score"] for r in rs], [tuple(r["summary"]["score_ci"]) for r in rs], fmt))
        rows.append(row)
    return _render(["Agent", "Track", *splits], rows, fmt,
                   "Bench Score (success % averaged over tasks) per split")


def tasks_table(results: list[dict], split: str = "test", fmt: str = "md") -> str:
    rows = []
    for (label, track), by_split in _group(results).items():
        rs = by_split.get(split, [])
        if not rs:
            continue
        row = [label, track]
        for task in TASK_ORDER:
            vals = [r["summary"]["by_task"][task]["success"] for r in rs if task in r["summary"]["by_task"]]
            row.append(f"{100 * np.mean(vals):.0f}" if vals else "–")
        row.append(_cell([r["summary"]["score"] for r in rs], [tuple(r["summary"]["score_ci"]) for r in rs], fmt))
        rows.append(row)
    header = ["Agent", "Track", *[f"{t} (L{TASKS[t].level})" for t in TASK_ORDER], "Score"]
    return _render(header, rows, fmt, f"Success % per task on `{split}`")


def levels_table(results: list[dict], split: str = "test", fmt: str = "md") -> str:
    levels = sorted({t.level for t in TASKS.values()})
    rows = []
    for (label, track), by_split in _group(results).items():
        rs = by_split.get(split, [])
        if not rs:
            continue
        row = [label, track]
        for level in levels:
            vals = [r["summary"]["by_level"].get(str(level)) for r in rs]
            vals = [v for v in vals if v is not None]
            row.append(f"{100 * np.mean(vals):.0f}" if vals else "–")
        rows.append(row)
    return _render(["Agent", "Track", *[f"L{lv}" for lv in levels]], rows, fmt,
                   f"Success % per difficulty level on `{split}`")


def errors_table(results: list[dict], split: str = "test", fmt: str = "md") -> str:
    rows = []
    for (label, track), by_split in _group(results).items():
        rs = by_split.get(split, [])
        if not rs:
            continue
        mean = lambda key: np.mean([r["summary"].get(key, np.nan) for r in rs])
        times = [t["time"] for r in rs for t in r["summary"]["by_task"].values() if t["time"] is not None]
        rows.append([label, track, f"{mean('wrong_keys'):.2f}", f"{mean('failed_clicks'):.2f}",
                     f"{mean('collisions'):.2f}", f"{mean('failed_contacts'):.2f}",
                     f"{np.mean(times):.1f}" if times else "–"])
    return _render(["Agent", "Track", "Wrong keys", "Failed clicks", "Collisions", "Failed contacts",
                    "Time to success (s)"], rows, fmt, f"Errors per episode and mean time to success on `{split}`")


def report(results: list[dict], table: str = "splits", split: str = "test", fmt: str = "md") -> str:
    if table == "splits":
        return splits_table(results, fmt)
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return {"tasks": tasks_table, "levels": levels_table, "errors": errors_table}[table](results, split, fmt)
