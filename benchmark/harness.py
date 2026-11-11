"""Evaluation harness: run an agent on a split and score it.

    from benchmark.harness import evaluate
    result = evaluate("oracle", "test", workers=8, out="results/oracle_test.json")

Each worker process builds its own simulator and agent; episodes are distributed over the
workers and the per-episode records are merged in episode order, so the result does not
depend on the number of workers. The result file holds every episode record, the summary
(benchmark/metrics.py) and the metadata needed to reproduce it.
"""

from __future__ import annotations

import json
import multiprocessing
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import environment  # noqa: F401  (MUJOCO_GL / pygame env vars before mujoco is imported)
from benchmark import BENCHMARK_NAME, BENCHMARK_VERSION
from benchmark.agents import Agent, agent_class, default_name, make_agent
from benchmark.metrics import summarize
from benchmark.suite import EPISODES_PER_TASK, SPLITS, EpisodeSpec, episodes
from benchmark.tracks import Track, get_track
from environment.computer_env import ComputerUseEnv


def run_episode(env: ComputerUseEnv, agent: Agent, spec: EpisodeSpec, track: Track) -> dict:
    task = spec.make_task()
    obs = env.reset(task, variation=spec.variation(), dynamics=spec.hidden_dynamics())
    agent.reset(task.instruction, spec.seed)
    total, done = 0.0, False
    while not done:
        obs, reward, done, _ = env.step(agent.act(track.filter(obs)))
        total += reward
    m = env.metrics()
    return {
        **spec.to_dict(),
        "success": bool(m["success"]),
        "time": float(m["time"]),
        "steps": int(m["steps"]),
        "return": float(total),
        "progress": float(m["progress"]),
        **{k: int(m[k]) for k in ("wrong_keys", "failed_clicks", "collisions", "failed_contacts", "key_presses",
                                   "clicks")},
    }


# --------------------------------------------------------------------- worker processes
_WORKER: dict = {}


def _init_worker(agent_spec: str, track_name: str, physics: str = "nominal") -> None:
    track = get_track(track_name)
    config = track.env_config()
    config.physics = physics
    env = ComputerUseEnv(config)
    _WORKER.update(env=env, agent=make_agent(agent_spec, env), track=track)


def _run_spec(spec: dict) -> dict:
    return run_episode(_WORKER["env"], _WORKER["agent"], EpisodeSpec(**spec), _WORKER["track"])


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def metadata(workers: int) -> dict:
    import mujoco
    import pygame

    return {
        "benchmark": BENCHMARK_NAME,
        "version": BENCHMARK_VERSION,
        "git_commit": _git("rev-parse", "HEAD") or None,
        # Uncommitted changes to tracked code mean git_commit alone does not reproduce the result.
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "pygame": pygame.version.ver,
        "platform": platform.platform(),
        "workers": workers,
    }


def evaluate(agent_spec: str, split: str, track: str | None = None, tasks=None, per_task: int = EPISODES_PER_TASK,
             workers: int = 1, out: str | Path | None = None, verbose: bool = True, name: str | None = None) -> dict:
    """Evaluate an agent on a split; returns (and optionally writes) the result dict. `name`
    groups results in reports (e.g. the seeds of one recipe); default: default_name(agent)."""
    cls, _ = agent_class(agent_spec)
    track_name = track or cls.track
    physics = SPLITS[split].physics
    specs = episodes(split, tasks, per_task)
    t0 = time.time()
    records: list[dict] = []

    def report(rec):
        records.append(rec)
        if verbose and (len(records) % 20 == 0 or len(records) == len(specs)):
            rate = np.mean([r["success"] for r in records])
            print(f"  [{split}] {len(records):4d}/{len(specs)} episodes  success so far {100 * rate:5.1f}%  "
                  f"{time.time() - t0:6.1f}s", flush=True)

    if workers <= 1:
        _init_worker(agent_spec, track_name, physics)
        for spec in specs:
            report(_run_spec(spec.to_dict()))
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(agent_spec, track_name, physics)) as pool:
            for rec in pool.imap_unordered(_run_spec, [s.to_dict() for s in specs], chunksize=1):
                report(rec)

    records.sort(key=lambda r: r["episode_id"])
    result = {
        "benchmark": BENCHMARK_NAME,
        "version": BENCHMARK_VERSION,
        "split": split,
        "physics": physics,
        "track": track_name,
        "agent": agent_spec,
        "agent_name": name or default_name(agent_spec),
        "privileged": bool(cls.privileged),
        "per_task": per_task,
        "summary": summarize(records),
        "episodes": records,
        "meta": {**metadata(workers), "wall_seconds": round(time.time() - t0, 1)},
    }
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1))
        if verbose:
            print(f"  saved {out}")
    return result
