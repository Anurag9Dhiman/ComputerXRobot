"""Results figure for BENCHMARK.md / a paper (needs matplotlib).

    python -m benchmark.plot results/*.json --agent bc --out docs/benchmark_results.png

Left: Bench Score of the learned agent on each test split (mean over training seeds, error
bars = std), with the privileged oracle as a labelled reference line and the random agent's
best score in the title.
Right: its success per difficulty level on each split.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from benchmark.report import load
from benchmark.suite import TEST_SPLITS

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"  # reference palette, categorical slot 1 (validated)
LEVEL_RAMP = ["#a9cdf5", "#6aa6eb", "#2a78d6", "#18519a"]  # one hue, light -> dark


def _scores(results, agent):
    by_split = defaultdict(list)
    levels = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r["agent_name"] == agent:
            by_split[r["split"]].append(r["summary"]["score"])
            for lv, v in r["summary"]["by_level"].items():
                levels[r["split"]][lv].append(v)
    return by_split, levels


def plot(results: list[dict], agent: str, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
                         "ytick.color": INK_2, "text.color": INK, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE})
    splits = [s for s in TEST_SPLITS if any(r["split"] == s for r in results)]
    by_split, levels = _scores(results, agent)
    refs = {name: [np.mean([r["summary"]["score"] for r in results if r["agent_name"] == name and r["split"] == s])
                   for s in splits] for name in ("oracle", "random")}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.35]})
    x = np.arange(len(splits))
    mean = np.array([100 * np.mean(by_split[s]) for s in splits])
    std = np.array([100 * np.std(by_split[s]) for s in splits])
    ax.bar(x, mean, width=0.55, color=SERIES, zorder=3)
    ax.errorbar(x, mean, yerr=std, fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=4)
    for xi, m in zip(x, mean):  # values inside the bars, clear of the error bars and reference lines
        ax.text(xi, m / 2, f"{m:.0f}", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
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
    ax.set_title(f"{agent}, state track: Bench Score per split{random_note}", fontsize=10.5, loc="left")

    lv_names = sorted({lv for s in splits for lv in levels[s]})
    width = 0.8 / max(len(lv_names), 1)
    for i, lv in enumerate(lv_names):
        vals = [100 * np.mean(levels[s][lv]) for s in splits]
        bx.bar(x + (i - (len(lv_names) - 1) / 2) * width, vals, width=width * 0.9,
               color=LEVEL_RAMP[i % len(LEVEL_RAMP)], label=f"L{lv}", zorder=3)
    bx.set_xticks(x, splits)
    bx.set_ylim(0, 105)
    bx.set_ylabel("Success (%)")
    bx.set_title(f"{agent}: success per difficulty level", fontsize=10.5, loc="left")
    bx.legend(ncols=len(lv_names), frameon=False, loc="lower right", fontsize=8.5, bbox_to_anchor=(1, 1.0),
              handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)

    for a in (ax, bx):
        a.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+")
    p.add_argument("--agent", default="bc")
    p.add_argument("--out", default="docs/benchmark_results.png")
    args = p.parse_args(argv)
    plot(load(args.results), args.agent, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
