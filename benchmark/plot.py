"""Results figures for BENCHMARK.md / a paper (needs matplotlib).

    python -m benchmark.plot results/*.json --agent bc --out docs/benchmark_results.png
    python -m benchmark.plot results/*.json --hard --out docs/hard_results.png
    python -m benchmark.plot results/*.json --vision --out docs/vision_results.png

Standard splits. Left: Bench Score of the learned agent on each test split (mean over
training seeds; error bars: 95% bootstrap interval over seeds and episodes, benchmark/stats.py),
with the privileged oracle as a labelled reference line and the random agent's best score in
the title. Right: its success per difficulty level on each split.

Hard tier (--hard). Left: Bench Score of every learned agent on hard_test and hard_test_ood.
Right: the dynamics ablation (benchmark/ablation.py), one row per condition. Each agent keeps
one colour in both panels.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from benchmark.report import load
from benchmark.stats import aggregate
from benchmark.suite import TEST_SPLITS

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"  # reference palette, categorical slot 1 (validated)
LEVEL_RAMP = ["#a9cdf5", "#6aa6eb", "#2a78d6", "#18519a"]  # one hue, light -> dark
# Hard-tier agents in a fixed order, one categorical slot each (reference palette, validated;
# slots 3-5 are below 3:1 on the surface, so every value is also labelled).
HARD_AGENTS = ("bc-nominal", "bc", "bc-history", "bc-dagger", "bc-history-dagger", "bc-gru", "bc-gru-dagger")
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")


def _scores(results, agent):
    by_split = defaultdict(list)
    levels = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r["agent_name"] == agent:
            by_split[r["split"]].append(r["summary"]["score"])
            for lv, v in r["summary"]["by_level"].items():
                levels[r["split"]][lv].append(v)
    return by_split, levels


def _mean_ci(runs: list[dict]) -> tuple[float, float, float]:
    """Mean score (%) of an agent's runs on one split and its 95% interval (benchmark/stats.py)."""
    a = aggregate(runs)
    return 100 * a["mean"], 100 * a["mean_ci"][0], 100 * a["mean_ci"][1]


def _bars_with_ci(ax, xs, stats, color, width, label=None, inside=False):
    """Bars at the means with asymmetric 95% interval whiskers; values above (or inside) the bars."""
    means = np.array([m for m, _, _ in stats])
    err = np.array([[m - lo for m, lo, _ in stats], [hi - m for m, _, hi in stats]])
    ax.bar(xs, means, width=width, color=color, edgecolor=SURFACE, linewidth=1.5, label=label, zorder=3)
    ax.errorbar(xs, means, yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=2.5, zorder=4)
    for x, (m, _, hi) in zip(xs, stats):
        if inside:
            ax.text(x, m / 2, f"{m:.0f}", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        else:  # above the whisker, but clear of the reference line at 100
            y = hi + 1.5
            ax.text(x, 102.5 if 95 < y < 102.5 else y, f"{m:.0f}", ha="center", va="bottom", fontsize=8, color=INK_2)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
                         "ytick.color": INK_2, "text.color": INK, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE})
    return plt


def _style(*axes) -> None:
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(length=0)


def plot(results: list[dict], agent: str, out: str) -> None:
    plt = _pyplot()
    results = [r for r in results if "summary" in r]
    splits = [s for s in TEST_SPLITS if any(r["split"] == s for r in results)]
    by_split, levels = _scores(results, agent)
    refs = {name: [np.mean([r["summary"]["score"] for r in results if r["agent_name"] == name and r["split"] == s])
                   for s in splits] for name in ("oracle", "random")}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.35]})
    x = np.arange(len(splits))
    runs = {s: [r for r in results if r["agent_name"] == agent and r["split"] == s] for s in splits}
    # Values inside the bars, clear of the whiskers and the reference line.
    _bars_with_ci(ax, x, [_mean_ci(runs[s]) for s in splits], SERIES, 0.55, inside=True)
    n_runs = max(len(v) for v in runs.values())
    oracle = 100 * np.array(refs["oracle"])
    if np.all(np.isfinite(oracle)):
        ax.plot([-0.45, len(splits) - 0.55], [oracle.mean()] * 2, "--", color=INK_2, lw=1.2, zorder=2)
        ax.text(len(splits) - 0.5, oracle.mean() + 2, "oracle (privileged)", ha="right", va="bottom",
                fontsize=8.5, color=INK_2)
    random = 100 * np.array(refs["random"])
    random_note = f" (random ≤ {np.nanmax(random):.1f})" if np.all(np.isfinite(random)) else ""
    ax.set_xticks(x, splits)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Bench Score (%)")
    ax.set_title(f"{agent}, state track: Bench Score per split\nmean of {n_runs} seeds, 95% CI{random_note}",
                 fontsize=10.5, loc="left")

    lv_names = sorted({lv for s in splits for lv in levels[s]})
    width = 0.8 / max(len(lv_names), 1)
    for i, lv in enumerate(lv_names):
        vals = [100 * np.mean(levels[s][lv]) for s in splits]
        bx.bar(x + (i - (len(lv_names) - 1) / 2) * width, vals, width=width * 0.9,
               color=LEVEL_RAMP[i % len(LEVEL_RAMP)], label=f"L{lv}", zorder=3)
    bx.set_xticks(x, splits)
    bx.set_ylim(0, 105)
    bx.set_ylabel("Success (%)")
    bx.set_title(f"{agent}: success per difficulty level\n", fontsize=10.5, loc="left")
    bx.legend(ncols=len(lv_names), frameon=False, loc="lower right", fontsize=8.5, bbox_to_anchor=(1, 1.0),
              handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)

    for a in (ax, bx):
        a.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    _style(ax, bx)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


def plot_hard(results: list[dict], out: str) -> None:
    from benchmark.ablation import CONDITIONS
    from benchmark.suite import HARD_TEST_SPLITS

    plt = _pyplot()
    evals = [r for r in results if "summary" in r and r["split"] in HARD_TEST_SPLITS]
    ablations = [r for r in results if r.get("analysis") == "dynamics_ablation" and not r["privileged"]]
    names = {r["agent_name"] for r in evals + ablations}
    agents = [a for a in HARD_AGENTS if a in names]
    color = {a: CATEGORICAL[HARD_AGENTS.index(a)] for a in agents}
    random = [r["summary"]["score"] for r in evals if r["agent_name"] == "random"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    splits = [s for s in HARD_TEST_SPLITS if any(r["split"] == s for r in evals)]
    width = 0.8 / max(len(agents), 1)
    n_runs = 0
    for i, a in enumerate(agents):
        xs, stats = [], []
        for j, s in enumerate(splits):
            rs = [r for r in evals if r["agent_name"] == a and r["split"] == s]
            if rs:
                xs.append(j + (i - (len(agents) - 1) / 2) * width)
                stats.append(_mean_ci(rs))
                n_runs = max(n_runs, len(rs))
        _bars_with_ci(ax, xs, stats, color[a], width, label=a)
    ax.plot([-0.45, len(splits) - 0.55], [100, 100], "--", color=INK_2, lw=1.2, zorder=2)
    # The non-privileged adaptive expert sits on the oracle's line; name it there.
    adaptive = [100 * r["summary"]["score"] for r in evals if r["agent_name"] == "adaptive-expert"]
    ref = "oracle (privileged)" + (f"\nadaptive-expert ≥ {min(adaptive):.1f}" if adaptive else "")
    ax.text(len(splits) - 0.55, 98, ref, ha="right", va="top", fontsize=8.5, color=INK_2, linespacing=1.4)
    ax.set_xticks(range(len(splits)), splits)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Bench Score (%)")
    note = f" (random ≤ {100 * max(random):.1f})" if random else ""
    ax.set_title(f"Hard tier: Bench Score, mean of {n_runs} seeds, 95% CI{note}", fontsize=10.5, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)

    # Dot plot of seed means, one sub-row per agent so dots never overlap. Values and seed std
    # are in the ablation table (`report --table ablation`), which keeps the panel uncluttered.
    rows = np.arange(len(CONDITIONS))
    abl_agents = [a for a in agents if any(r["agent_name"] == a for r in ablations)]
    offset = 0.6 / max(len(abl_agents), 1)
    for i, a in enumerate(abl_agents):
        runs = [r for r in ablations if r["agent_name"] == a]
        m = np.array([np.mean([100 * r["conditions"][c]["score"] for r in runs]) for c in CONDITIONS])
        y = rows + (i - (len(abl_agents) - 1) / 2) * offset
        bx.hlines(y, 0, m, color=GRID, lw=0.8, zorder=2)
        bx.scatter(m, y, s=40, color=color[a], edgecolors=SURFACE, linewidths=1.2, zorder=4)
    bx.set_yticks(rows, CONDITIONS)
    bx.invert_yaxis()
    bx.set_xlim(0, 104)
    bx.set_xlabel("Bench Score (%)")
    bx.axhline(0.5, color=GRID, lw=0.8)
    bx.axhline(len(CONDITIONS) - 1.5, color=GRID, lw=0.8)
    bx.set_title("Dynamics ablation (hard_test): one parameter out of range", fontsize=10.5,
                 loc="left")
    bx.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    _style(ax, bx)
    fig.legend(*ax.get_legend_handles_labels(), ncols=len(agents), frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), fontsize=9, handlelength=1.2, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


VISION_AGENTS = (("bc", "state"), ("vision-bc-subgoal", "vision_subgoal"), ("vision-bc", "vision"))


def plot_vision(results: list[dict], out: str) -> None:
    """Left: Bench Score per test split, state-track bc against vision-bc (same demonstrations).
    Right: success per task on `test` for bc (state, sub-goal given), vision-bc-subgoal (pixels
    plus sub-goal; diagnostic) and vision-bc (pixels only)."""
    from benchmark.suite import TASK_ORDER

    plt = _pyplot()
    results = [r for r in results if "summary" in r]
    present = [(a, t) for a, t in VISION_AGENTS if any(r["agent_name"] == a and r["track"] == t for r in results)]
    color = {a: CATEGORICAL[i] for i, (a, _) in enumerate(VISION_AGENTS)}
    label = {"bc": "bc (state track)", "vision-bc-subgoal": "vision-bc + sub-goal (diagnostic)",
             "vision-bc": "vision-bc (vision track)"}

    def runs(agent, track, split):
        return [r for r in results if r["agent_name"] == agent and r["track"] == track and r["split"] == split]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.6]})
    split_agents = [(a, t) for a, t in present if a != "vision-bc-subgoal"]
    width = 0.8 / max(len(split_agents), 1)
    for i, (a, t) in enumerate(split_agents):
        xs, stats = [], []
        for j, split in enumerate(TEST_SPLITS):
            if runs(a, t, split):
                xs.append(j + (i - (len(split_agents) - 1) / 2) * width)
                stats.append(_mean_ci(runs(a, t, split)))
        _bars_with_ci(ax, xs, stats, color[a], width)
    ax.set_xticks(range(len(TEST_SPLITS)), TEST_SPLITS)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Bench Score (%)")
    ax.set_title("Bench Score per test split, mean over seeds, 95% CI", fontsize=10.5, loc="left")

    width = 0.8 / max(len(present), 1)
    for i, (a, t) in enumerate(present):
        rs = runs(a, t, "test")
        vals = [100 * np.mean([r["summary"]["by_task"][task]["success"] for r in rs]) for task in TASK_ORDER]
        bx.bar(np.arange(len(TASK_ORDER)) + (i - (len(present) - 1) / 2) * width, vals, width=width,
               color=color[a], edgecolor=SURFACE, linewidth=1.2, label=label[a], zorder=3)
    bx.set_xticks(range(len(TASK_ORDER)), [t.replace("_", "\n") for t in TASK_ORDER], fontsize=8.5)
    bx.set_ylim(0, 105)
    bx.set_ylabel("Success (%)")
    bx.set_title("Success per task on test (mean over seeds)", fontsize=10.5, loc="left")
    for a in (ax, bx):
        a.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    _style(ax, bx)
    fig.legend(*bx.get_legend_handles_labels(), ncols=len(present), frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), fontsize=9, handlelength=1.2, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+")
    p.add_argument("--agent", default="bc")
    p.add_argument("--hard", action="store_true", help="hard-tier figure (all learned agents, dynamics ablation)")
    p.add_argument("--vision", action="store_true", help="vision-track figure (bc, vision-bc, vision-bc-subgoal)")
    p.add_argument("--out", default="docs/benchmark_results.png")
    args = p.parse_args(argv)
    if args.hard:
        plot_hard(load(args.results), args.out)
    elif args.vision:
        plot_vision(load(args.results), args.out)
    else:
        plot(load(args.results), args.agent, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
