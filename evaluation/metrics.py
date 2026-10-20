"""Aggregate episode results into the evaluation metrics.

    success rate                   fraction of episodes that met the task's success check
    time to complete               mean simulated seconds, successful episodes only
    wrong key presses              key presses that were not the character the task needed
    failed clicks                  clicks that hit nothing or the wrong widget
    collisions                     robot contacts with anything but key tops / mouse surfaces
    failed contacts                finger touched a key or button without actuating it
Splits: "train" content was seen in demonstrations, "test" content was not (unseen tasks).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

COLUMNS = ("episodes", "success_rate", "mean_time", "wrong_keys", "failed_clicks", "collisions",
           "failed_contacts")


def summarize(results: list[dict]) -> dict:
    if not results:
        return {c: 0 for c in COLUMNS}
    success = np.array([r["success"] for r in results])
    times = [r["time"] for r in results if r["success"]]
    per_ep = lambda key: float(np.mean([r[key] for r in results]))
    return {
        "episodes": len(results),
        "success_rate": float(success.mean()),
        "mean_time": float(np.mean(times)) if times else float("nan"),
        "wrong_keys": per_ep("wrong_keys"),
        "failed_clicks": per_ep("failed_clicks"),
        "collisions": per_ep("collisions"),
        "failed_contacts": per_ep("failed_contacts"),
    }


def summarize_by(results: list[dict], *keys: str) -> dict[tuple, dict]:
    groups: dict[tuple, list] = defaultdict(list)
    for r in results:
        groups[tuple(r.get(k) for k in keys)].append(r)
    return {k: summarize(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def format_table(summaries: dict[tuple, dict], key_names: tuple[str, ...]) -> str:
    header = [*key_names, "eps", "success", "time(s)", "wrong keys", "failed clicks", "collisions",
              "failed contacts"]
    rows = []
    for key, s in summaries.items():
        rows.append([
            *[str(k) for k in key],
            str(s["episodes"]),
            f"{100 * s['success_rate']:.0f}%",
            f"{s['mean_time']:.1f}",
            f"{s['wrong_keys']:.2f}",
            f"{s['failed_clicks']:.2f}",
            f"{s['collisions']:.2f}",
            f"{s['failed_contacts']:.2f}",
        ])
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))
    return "\n".join([line(header), line(["-" * w for w in widths]), *map(line, rows)])


def save_results(results: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episodes": results,
        "summary": {"/".join(map(str, k)): v for k, v in summarize_by(results, "task", "split").items()},
    }
    path.write_text(json.dumps(payload, indent=2, default=float))
