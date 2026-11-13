"""Single-factor dynamics ablation: which hidden dynamics break an agent?

Every condition replays the same `hard_test` episodes (content, scene, seed). "in range" keeps
each episode's hidden dynamics as drawn (`real_id`). Each factor condition pushes exactly one
hidden parameter to the value it takes in a `real_ood` sample drawn for that episode, and
keeps all other parameters in range. "all" pushes every parameter, as in `hard_test_ood`. The
drop from "in range" isolates what each real-world effect costs the agent.

    python -m benchmark ablate --agent bc:checkpoints/bc_s0.pt --name bc-nominal --workers 8 \\
        --out results/ablation_nominal_bc_s0.json
    python -m benchmark report results/ablation_*.json --table ablation

This is an analysis tool, not an official split: Bench Scores are reported on the splits.
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np

from benchmark import BENCHMARK_NAME, BENCHMARK_VERSION
from benchmark.harness import _WORKER, _init_worker, metadata, run_episode
from benchmark.metrics import summarize
from benchmark.suite import EpisodeSpec, _seed, episodes
from environment.dynamics import HiddenDynamics, sample_dynamics
from environment.keyboard_layout import KEYS

FACTORS: dict[str, tuple[str, ...]] = {  # condition -> HiddenDynamics fields pushed out of range
    "control latency": ("action_delay",),
    "display latency": ("display_latency",),
    "actuation noise": ("action_noise",),
    "observation noise": ("obs_noise",),
    "arm sag": ("gravcomp",),
    "pointer acceleration": ("pointer_accel",),
    "mouse rotation": ("mouse_yaw",),
    "key spring force": ("key_stiffness",),
    "key actuation depth": ("key_actuation",),
    "auto-repeat": ("autorepeat_delay",),
    "double-click time": ("double_click_time",),
}
ALL_FIELDS = tuple(f for fields in FACTORS.values() for f in fields)
CONDITIONS = ("in range", *FACTORS, "all")
SPLIT = "hard_test"


def condition_dynamics(spec: EpisodeSpec, condition: str) -> HiddenDynamics:
    base = spec.hidden_dynamics()
    if condition == "in range":
        return base
    fields = ALL_FIELDS if condition == "all" else FACTORS[condition]
    ood = sample_dynamics("real_ood", np.random.default_rng(_seed(spec.seed, "ablation")), len(KEYS))
    return dataclasses.replace(base, **{f: getattr(ood, f) for f in fields}, name=f"ablation:{condition}")


def _run_job(job: tuple[dict, str]) -> dict:
    spec, condition = EpisodeSpec(**job[0]), job[1]
    rec = run_episode(_WORKER["env"], _WORKER["agent"], spec, _WORKER["track"],
                      dynamics=condition_dynamics(spec, condition))
    return {**rec, "condition": condition}


def ablate(agent_spec: str, per_task: int = 10, workers: int = 1, out: str | Path | None = None,
           name: str | None = None, conditions=CONDITIONS, verbose: bool = True) -> dict:
    from benchmark.agents import agent_class, default_name

    cls, _ = agent_class(agent_spec)
    jobs = [(s.to_dict(), c) for c in conditions for s in episodes(SPLIT, per_task=per_task)]
    t0 = time.time()
    if workers <= 1:
        _init_worker(agent_spec, cls.track, "realistic")
        records = [_run_job(j) for j in jobs]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(agent_spec, cls.track, "realistic")) as pool:
            records = pool.map(_run_job, jobs, chunksize=1)
    by_condition = {c: summarize([r for r in records if r["condition"] == c]) for c in conditions}
    result = {
        "benchmark": BENCHMARK_NAME,
        "version": BENCHMARK_VERSION,
        "analysis": "dynamics_ablation",
        "split": SPLIT,
        "agent": agent_spec,
        "agent_name": name or default_name(agent_spec),
        "privileged": bool(cls.privileged),
        "episodes_per_task": per_task,
        "conditions": by_condition,
        "episodes": records,
        "meta": {**metadata(workers), "wall_time_s": round(time.time() - t0, 1)},
    }
    if verbose:
        for c, s in by_condition.items():
            print(f"  {c:22s} {100 * s['score']:5.1f}")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=1))
        if verbose:
            print(f"  saved {out}")
    return result
