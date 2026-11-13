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

from benchmark.suite import HARD_TEST_SPLITS, SPLITS, TASK_ORDER, TEST_SPLITS
from environment.task_manager import TASKS

TABLES = ("splits", "hard", "tasks", "levels", "errors", "ablation")


def load(paths) -> list[dict]:
    """Evaluation and ablation results among `paths`; other files, such as the
    dynamics-prediction track's results/world_model.json, are skipped so that `results/*.json`
    works."""
    results = [json.loads(Path(p).read_text()) for p in paths]
    return [r for r in results if "per_task" in r or r.get("analysis") == "dynamics_ablation"]


def ablation_table(results: list[dict], fmt: str = "md") -> str:
    """Single-factor dynamics ablation (benchmark/ablation.py): one row per condition, one
    column per agent (mean ± std over runs with the same name)."""
    from benchmark.ablation import CONDITIONS

    by_agent: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_agent[_label(r)].append(r)
    agents = sorted(by_agent, key=lambda a: ("privileged" in a, a))
    rows = []
    for c in CONDITIONS:
        row = [c]
        for a in agents:
            rs = [r for r in by_agent[a] if c in r["conditions"]]
            row.append(_cell([r["conditions"][c]["score"] for r in rs],
                             [tuple(r["conditions"][c]["score_ci"]) for r in rs], fmt))
        rows.append(row)
    n = {r["episodes_per_task"] for r in results}
    return _render(["Condition", *agents], rows, fmt,
                   f"Dynamics ablation on hard_test ({'/'.join(map(str, sorted(n)))} episodes per task): "
                   "one hidden parameter pushed beyond its training range")


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


def splits_table(results: list[dict], fmt: str = "md", hard: bool = False) -> str:
    candidates = (*HARD_TEST_SPLITS, "hard_val") if hard else (*TEST_SPLITS, "val", "train")
    splits = [s for s in candidates if any(r["split"] == s for r in results)]
    rows = []
    for (label, track), by_split in _group(results).items():
        if not any(by_split.get(s) for s in splits):
            continue
        row = [label, track]
        for s in splits:
            rs = by_split.get(s, [])
            row.append(_cell([r["summary"]["score"] for r in rs], [tuple(r["summary"]["score_ci"]) for r in rs], fmt))
        rows.append(row)
    title = "Hard tier (realistic physics, hidden dynamics)" if hard else "Bench Score"
    return _render(["Agent", "Track", *splits], rows, fmt, f"{title}: success % averaged over tasks, per split")


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
    ablations = [r for r in results if r.get("analysis") == "dynamics_ablation"]
    if table == "ablation":
        return ablation_table(ablations, fmt)
    results = [r for r in results if r.get("analysis") != "dynamics_ablation"]
    if table in ("splits", "hard"):
        return splits_table(results, fmt, hard=table == "hard")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return {"tasks": tasks_table, "levels": levels_table, "errors": errors_table}[table](results, split, fmt)
