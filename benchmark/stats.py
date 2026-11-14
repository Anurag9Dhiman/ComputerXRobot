"""Interval estimates over training seeds and test episodes.

A learned baseline's score varies with its training seed and with the finite set of test
episodes. Following Agarwal et al., "Deep Reinforcement Learning at the Edge of the
Statistical Precipice" (NeurIPS 2021), results are reported as interval estimates of robust
aggregates rather than as a mean over a handful of seeds:

    mean     Bench Score (success averaged over tasks), averaged over seeds
    IQM      interquartile mean of the per-(seed, task) success rates: the mean of their
             middle 50%, robust to one seed that collapses or a task every seed solves
    P(X>Y)   probability of improvement: for a random task, the probability that a random
             seed of X has a higher success rate than a random seed of Y (ties count half),
             averaged over tasks
    Δ        difference of the mean scores, X - Y

95% intervals come from a bootstrap that resamples both sources of variation: the seeds with
replacement, and the episodes of each task with replacement (stratified by task, as
`metrics.score_ci`). Every seed plays the same test episodes (seeds and episodes are crossed,
not nested), so one episode draw is shared by all seeds of a resample. With a single run, e.g.
a deterministic agent, this is the stratified episode bootstrap of `metrics.score_ci`. All
agents play the same episodes too, so a comparison shares the episode draw between both
agents: a paired bootstrap, tighter than comparing two independent intervals.

    from benchmark.stats import aggregate, compare
    aggregate(runs)          # runs: result dicts of one agent on one split (one per seed)
    compare(runs_x, runs_y)  # P(X > Y) and Δ with 95% intervals
"""

from __future__ import annotations

import numpy as np

N_BOOTSTRAP = 2000
ALPHA = 0.05


def success_matrix(runs: list[dict], condition: str | None = None) -> tuple[list[str], list[np.ndarray]]:
    """The episode ids (sorted) and per task, in id order, a (runs, episodes) array of 0/1
    successes. Every run must have played the same episodes; `condition` selects one condition
    of an ablation result."""
    if not runs:
        raise ValueError("no runs")
    by_run = []
    for r in runs:
        eps = [e for e in r["episodes"] if condition is None or e.get("condition") == condition]
        by_run.append({e["episode_id"]: float(e["success"]) for e in eps})
    ids = sorted(by_run[0])
    if any(sorted(b) != ids for b in by_run[1:]):
        raise ValueError("runs cover different episodes; compare runs on the same split")
    tasks = sorted({i.split("/")[1] for i in ids})
    mats = []
    for t in tasks:
        task_ids = [i for i in ids if i.split("/")[1] == t]
        mats.append(np.array([[b[i] for i in task_ids] for b in by_run]))
    return ids, mats


def _iqm(rates: np.ndarray) -> np.ndarray:
    """Interquartile mean over the last two axes (seeds, tasks), as scipy.stats.trim_mean(x, 0.25)."""
    flat = np.sort(rates.reshape(*rates.shape[:-2], -1), axis=-1)
    n = flat.shape[-1]
    k = int(0.25 * n)
    return flat[..., k:n - k].mean(axis=-1)


def _episode_draws(mats: list[np.ndarray], rng: np.random.Generator, n_boot: int) -> list[np.ndarray]:
    return [rng.integers(0, m.shape[1], (n_boot, m.shape[1])) for m in mats]


def _resample(mats: list[np.ndarray], rng: np.random.Generator, n_boot: int,
              episode_idx: list[np.ndarray]) -> np.ndarray:
    """(n_boot, seeds, tasks) success rates of bootstrap resamples: seeds drawn with
    replacement, episodes taken from `episode_idx` (one (n_boot, episodes) draw per task)."""
    n_runs = mats[0].shape[0]
    seeds = rng.integers(0, n_runs, (n_boot, n_runs))
    out = np.empty((n_boot, n_runs, len(mats)))
    for t, m in enumerate(mats):
        out[:, :, t] = m[seeds[:, :, None], episode_idx[t][:, None, :]].mean(axis=2)
    return out


def _interval(samples: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


def aggregate(runs: list[dict], condition: str | None = None, n_boot: int = N_BOOTSTRAP, seed: int = 0) -> dict:
    """Mean and IQM of one agent's runs (one per seed) on one split, with 95% intervals."""
    _, mats = success_matrix(runs, condition)
    rates = np.stack([m.mean(axis=1) for m in mats], axis=1)  # (seeds, tasks)
    rng = np.random.default_rng(seed)
    boot = _resample(mats, rng, n_boot, _episode_draws(mats, rng, n_boot))
    return {
        "runs": len(runs),
        "mean": float(rates.mean()),
        "mean_ci": _interval(boot.mean(axis=(1, 2))),
        "iqm": float(_iqm(rates)),
        "iqm_ci": _interval(_iqm(boot)),
        "seed_scores": [float(s) for s in rates.mean(axis=1)],
    }


def _prob_improvement(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """P(X > Y) from (..., seeds_x, tasks) and (..., seeds_y, tasks) success rates."""
    diff = x[..., :, None, :] - y[..., None, :, :]
    return ((diff > 0) + 0.5 * (diff == 0)).mean(axis=(-3, -2, -1))


def compare(runs_x: list[dict], runs_y: list[dict], condition: str | None = None, n_boot: int = N_BOOTSTRAP,
            seed: int = 0) -> dict:
    """X against Y on the same split: P(X > Y) and the score difference, with 95% intervals
    from a bootstrap that draws seeds per agent and episodes once for both (paired)."""
    ids_x, mx = success_matrix(runs_x, condition)
    ids_y, my = success_matrix(runs_y, condition)
    if ids_x != ids_y:
        raise ValueError("the agents played different episodes")
    rng = np.random.default_rng(seed)
    episode_idx = _episode_draws(mx, rng, n_boot)
    bx, by = _resample(mx, rng, n_boot, episode_idx), _resample(my, rng, n_boot, episode_idx)
    rx = np.stack([m.mean(axis=1) for m in mx], axis=1)
    ry = np.stack([m.mean(axis=1) for m in my], axis=1)
    return {
        "p_improvement": float(_prob_improvement(rx, ry)),
        "p_improvement_ci": _interval(_prob_improvement(bx, by)),
        "difference": float(rx.mean() - ry.mean()),
        "difference_ci": _interval(bx.mean(axis=(1, 2)) - by.mean(axis=(1, 2))),
    }
