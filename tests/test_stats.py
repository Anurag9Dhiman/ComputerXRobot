"""Interval estimates over seeds and episodes (benchmark/stats.py) and the report tables built on them."""

import numpy as np
import pytest

from benchmark.metrics import summarize
from benchmark.report import report
from benchmark.stats import _iqm, aggregate, compare

TASKS = ("clear_text", "open_app", "type_key")


def run(rates, name="agent", split="hard_test", episodes=10, seed=0):
    """A synthetic result: per task, `rate` of `episodes` episodes succeed (random which)."""
    rng = np.random.default_rng(seed)
    records = []
    for task, rate in zip(TASKS, rates):
        wins = np.zeros(episodes, bool)
        wins[rng.choice(episodes, int(round(rate * episodes)), replace=False)] = True
        for i, w in enumerate(wins):
            records.append({"episode_id": f"{split}/{task}/{i:03d}", "task": task, "level": 1, "success": bool(w),
                            "time": 1.0, "return": 0.0, "wrong_keys": 0, "failed_clicks": 0, "collisions": 0,
                            "failed_contacts": 0})
    return {"split": split, "track": "state", "agent_name": name, "privileged": False, "per_task": episodes,
            "summary": summarize(records), "episodes": records}


def test_single_run_interval_is_the_stratified_episode_bootstrap():
    r = run([0.3, 0.7, 0.5])
    a = aggregate([r])
    assert a["runs"] == 1 and a["mean"] == pytest.approx(r["summary"]["score"])
    assert a["mean_ci"] == pytest.approx(tuple(r["summary"]["score_ci"]))


def test_seed_variation_widens_the_interval():
    steady = [run([0.5, 0.5, 0.5], seed=s) for s in range(4)]
    erratic = [run([0.1] * 3, seed=0), run([0.9] * 3, seed=1), run([0.2] * 3, seed=2), run([0.8] * 3, seed=3)]
    a, b = aggregate(steady), aggregate(erratic)
    assert a["mean"] == pytest.approx(b["mean"]) == pytest.approx(0.5)
    width = lambda x: x["mean_ci"][1] - x["mean_ci"][0]
    assert width(b) > 2 * width(a)
    assert b["seed_scores"] == pytest.approx([0.1, 0.9, 0.2, 0.8])


def test_iqm_is_the_mean_of_the_middle_half():
    assert _iqm(np.arange(8.0).reshape(1, 8)) == pytest.approx(3.5)  # mean of 2, 3, 4, 5
    rates = np.array([[0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])  # one collapsed (seed, task)
    assert _iqm(rates) == 1.0 and rates.mean() < 1.0


def test_compare():
    x = [run([0.9, 0.8, 0.9], seed=s) for s in range(3)]
    y = [run([0.2, 0.1, 0.3], seed=s + 3) for s in range(3)]
    c = compare(x, y)
    assert c["p_improvement"] == 1.0 and c["p_improvement_ci"][0] > 0.9
    assert c["difference"] == pytest.approx(np.mean([0.9, 0.8, 0.9]) - np.mean([0.2, 0.1, 0.3]))
    same = compare(x, x)
    assert same["p_improvement"] == 0.5 and same["difference"] == 0.0
    assert same["difference_ci"][0] < 0 < same["difference_ci"][1]  # seeds are drawn per agent
    # Paired: both agents see the same episode draws, so one run against itself never differs.
    assert compare(x[:1], x[:1])["difference_ci"] == (0.0, 0.0)
    with pytest.raises(ValueError):
        compare(x, [run([0.5] * 3, split="hard_test_ood")])


def test_report_tables():
    results = [run([0.9, 0.8, 0.9], name="good", seed=s) for s in range(3)] + [run([0.2, 0.4, 0.3], name="weak")]
    agg = report(results, "aggregate", split="hard_test")
    assert "| good | state | 3 |" in agg and "| weak | state | 1 |" in agg
    assert "good vs weak" in report(results, "compare", split="hard_test")
    hard = report(results, "hard")
    assert "86.7 [" in hard  # mean [95% CI]
    assert "86.7 ± " in report(results, "hard", interval="std")
