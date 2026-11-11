"""Command line for ComputerXRobot-Bench.

    python -m benchmark list
    python -m benchmark run --agent oracle --all-test --workers 8
    python -m benchmark run --agent bc:checkpoints/bc_s0.pt --tag bc_s0 --all-test --workers 8
    python -m benchmark collect --per-task 40 --out data/train.npz
    python -m benchmark report results/*.json --table splits --format latex
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import environment  # noqa: F401  (MUJOCO_GL / pygame env vars)
from benchmark import BENCHMARK_NAME, BENCHMARK_VERSION
from benchmark.report import TABLES
from benchmark.suite import EPISODES_PER_TASK, HARD_TEST_SPLITS, SPLITS, TASK_ORDER, TEST_SPLITS
from benchmark.tracks import TRACKS
from environment.task_manager import SPLITS as CONTENT_SPLITS
from environment.task_manager import TASKS, make_task
from environment.variation import DISTRIBUTIONS


def cmd_list(args) -> int:
    print(f"{BENCHMARK_NAME} v{BENCHMARK_VERSION}\n")
    print("Tasks")
    for name in TASK_ORDER:
        cls, content = TASKS[name], CONTENT_SPLITS[name]
        example = make_task(name, content["test"][0])
        print(f"  L{cls.level} {name:13s} {example.time_limit:5.0f}s limit   train {len(content['train']):2d} / "
              f"test {len(content['test']):2d} items   e.g. {example.instruction}")
    print("\nSplits")
    for name, spec in SPLITS.items():
        print(f"  {name:14s} content={spec.content:5s} scene={spec.scene:12s} dynamics={spec.dynamics:8s} "
              f"{spec.description}")
    print(f"  ({EPISODES_PER_TASK} episodes per task per split -> {EPISODES_PER_TASK * len(TASKS)} per split)")
    print("\nTracks")
    for name, track in TRACKS.items():
        print(f"  {name:7s} {track.description}")
    print(f"\nScene distributions: {', '.join(DISTRIBUTIONS)}")
    return 0


def cmd_run(args) -> int:
    from benchmark.harness import evaluate

    groups = (list(TEST_SPLITS) if args.all_test else []) + (list(HARD_TEST_SPLITS) if args.all_hard else [])
    splits = groups + [s for s in args.split or [] if s not in groups]
    splits = splits or ["test"]
    for split in splits:
        tag = args.tag or re.sub(r"[^A-Za-z0-9_.-]+", "_", args.agent.split("/")[-1])
        out = Path(args.out_dir) / f"{tag}_{split}.json"
        print(f"{args.agent} on {split} ({args.per_task} episodes per task, {args.workers} workers)")
        result = evaluate(args.agent, split, track=args.track, tasks=args.tasks, per_task=args.per_task,
                          workers=args.workers, out=out, name=args.name)
        s = result["summary"]
        lo, hi = s["score_ci"]
        print(f"  Bench Score {100 * s['score']:.1f} (95% CI {100 * lo:.1f}-{100 * hi:.1f})   "
              f"by level {', '.join(f'L{k} {100 * v:.0f}' for k, v in s['by_level'].items())}\n")
    return 0


def cmd_collect(args) -> int:
    from benchmark.harness import collect

    if SPLITS[args.split].content != "train":
        sys.exit("demonstrations may only come from training splits (train, hard_train)")
    if args.workers > 1 and not args.student:
        sys.exit("--workers applies to DAgger collection (--student); oracle collection is sequential")
    n = len(TASK_ORDER if not args.tasks else args.tasks) * args.per_task
    who = f"student {args.student} labelled by the oracle (DAgger)" if args.student else f"oracle, noise {args.noise}"
    print(f"Collecting {n} episodes on {args.split} ({SPLITS[args.split].physics} physics, {who})")
    t0 = time.time()
    demos = collect(args.split, args.per_task, args.tasks, noise=args.noise, seed=args.seed, images=args.images,
                    student=args.student, workers=args.workers)
    demos.save(args.out)
    print(f"saved {len(demos)} samples from {len(demos.task_specs)}/{n} episodes to {args.out} "
          f"in {time.time() - t0:.0f}s")
    return 0


def cmd_ablate(args) -> int:
    from benchmark.ablation import ablate

    tag = args.tag or re.sub(r"[^A-Za-z0-9_.-]+", "_", args.agent.split("/")[-1])
    out = Path(args.out_dir) / f"ablation_{tag}.json"
    print(f"{args.agent}: dynamics ablation on hard_test ({args.per_task} episodes per task, {args.workers} workers)")
    ablate(args.agent, per_task=args.per_task, workers=args.workers, out=out, name=args.name)
    return 0


def cmd_report(args) -> int:
    from benchmark.report import load, report

    text = report(load(args.results), args.table, args.split, args.format)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmark", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="tasks, splits, tracks").set_defaults(func=cmd_list)

    p = sub.add_parser("run", help="evaluate an agent")
    p.add_argument("--agent", required=True, help="random | oracle | bc:<checkpoint> | <module>:<Class>")
    p.add_argument("--split", action="append", choices=list(SPLITS), help="repeatable (default: test)")
    p.add_argument("--all-test", action="store_true", help="all four test splits (plus any --split)")
    p.add_argument("--all-hard", action="store_true", help="hard_test and hard_test_ood (plus any --split)")
    p.add_argument("--track", choices=list(TRACKS), help="default: the agent's own track")
    p.add_argument("--tasks", nargs="+", choices=TASK_ORDER, help="subset of tasks (not a benchmark result)")
    p.add_argument("--per-task", type=int, default=EPISODES_PER_TASK,
                   help=f"episodes per task (official: {EPISODES_PER_TASK})")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--tag", help="name for the result files (default: from the agent spec)")
    p.add_argument("--name", help="agent name in result tables; runs with the same name are grouped as seeds")
    p.add_argument("--out-dir", default="results")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("collect", help="oracle demonstrations on a training split (for imitation learning)")
    p.add_argument("--split", default="train", choices=[s for s, v in SPLITS.items() if v.content == "train"])
    p.add_argument("--per-task", type=int, default=40)
    p.add_argument("--tasks", nargs="+", choices=TASK_ORDER)
    p.add_argument("--noise", type=float, default=0.5, help="perturbation size (fraction of max step)")
    p.add_argument("--images", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--student", metavar="CHECKPOINT",
                   help="DAgger: roll out this BC policy and let the oracle label the states it visits "
                        "(failed episodes are kept)")
    p.add_argument("--workers", type=int, default=1, help="parallel processes for DAgger collection")
    p.add_argument("--out", default="data/train.npz")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("ablate", help="single-factor dynamics ablation on hard_test (benchmark/ablation.py)")
    p.add_argument("--agent", required=True)
    p.add_argument("--per-task", type=int, default=10)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--name", help="agent name in the ablation table; runs with the same name are grouped")
    p.add_argument("--tag", help="output file tag (default: from the agent spec)")
    p.add_argument("--out-dir", default="results")
    p.set_defaults(func=cmd_ablate)

    p = sub.add_parser("report", help="Markdown / LaTeX tables from result files")
    p.add_argument("results", nargs="+")
    p.add_argument("--table", choices=TABLES, default="splits")
    p.add_argument("--split", default="test", help="split for the tasks / levels / errors tables")
    p.add_argument("--format", choices=("md", "latex"), default="md")
    p.add_argument("--out")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
