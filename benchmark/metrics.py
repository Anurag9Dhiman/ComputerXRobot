"""Scores and confidence intervals.

Per episode the harness records success, completion time, return and error counts. A split
is scored with:

    score        Bench Score: success rate averaged over tasks (each task weighs the same,
                 whatever its episode count), in [0, 1]
    score_ci     95% confidence interval of the score from a stratified bootstrap (episodes
                 resampled within each task, 2000 resamples)
    by_task      per task: success rate with 95% CI, mean time to success, mean errors
    by_level     per difficulty level: mean success of its tasks

Errors per episode: wrong key presses, failed clicks (nothing or the wrong widget hit),
collisions (robot touching anything but key tops and mouse surfaces), failed contacts
(finger touched a key or button without actuating it).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

N_BOOTSTRAP = 2000
ERROR_KEYS = ("wrong_keys", "failed_clicks", "collisions", "failed_contacts")


def bootstrap_ci(values, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    x = np.asarray(values, float)
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _stratified_score_ci(groups: list[np.ndarray], n_boot: int = N_BOOTSTRAP, alpha: float = 0.05,
                         seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    scores = np.zeros(n_boot)
    for g in groups:
        scores += g[rng.integers(0, len(g), (n_boot, len(g)))].mean(axis=1)
    scores /= len(groups)
    return float(np.quantile(scores, alpha / 2)), float(np.quantile(scores, 1 - alpha / 2))


def summarize(records: list[dict]) -> dict:
    by_task_records: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task_records[r["task"]].append(r)

    by_task, groups = {}, []
    for task, rs in by_task_records.items():
        success = np.array([float(r["success"]) for r in rs])
        groups.append(success)
        times = [r["time"] for r in rs if r["success"]]
        by_task[task] = {
            "level": rs[0]["level"],
            "episodes": len(rs),
            "success": float(success.mean()),
            "ci": bootstrap_ci(success),
            "time": float(np.mean(times)) if times else None,
            "return": float(np.mean([r["return"] for r in rs])),
            **{k: float(np.mean([r[k] for r in rs])) for k in ERROR_KEYS},
        }

    levels: dict[int, list[float]] = defaultdict(list)
    for t in by_task.values():
        levels[t["level"]].append(t["success"])

    return {
        "episodes": len(records),
        "score": float(np.mean([t["success"] for t in by_task.values()])) if by_task else float("nan"),
        "score_ci": _stratified_score_ci(groups) if groups else (float("nan"), float("nan")),
        "success_micro": float(np.mean([r["success"] for r in records])) if records else float("nan"),
        "by_task": by_task,
        "by_level": {str(level): float(np.mean(v)) for level, v in sorted(levels.items())},
        **({k: float(np.mean([r[k] for r in records])) for k in ERROR_KEYS} if records else {}),
    }
