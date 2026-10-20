"""Run policies on tasks and record per-episode results."""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from environment.computer_env import ComputerUseEnv, StepInfo
from environment.task_manager import SPLITS, Task, make_task


class Policy(Protocol):
    def reset(self) -> None: ...
    def act(self, obs: dict) -> np.ndarray: ...


StepCallback = Callable[[ComputerUseEnv, dict, StepInfo], None]


def run_episode(env: ComputerUseEnv, policy: Policy, task: Task, on_step: StepCallback | None = None) -> dict:
    """Run one episode until success or the task's time limit. Returns env.metrics()."""
    obs = env.reset(task)
    policy.reset()
    done = False
    while not done:
        obs, _, done, info = env.step(policy.act(obs))
        if on_step is not None:
            on_step(env, obs, info)
    return env.metrics()


def task_suite(names: list[str], split: str, episodes: int, seed: int = 0) -> list[Task]:
    """`episodes` tasks per task name, cycling through that split's strings in a shuffled order."""
    rng = np.random.default_rng(seed)
    tasks = []
    for name in names:
        options = list(SPLITS[name][split])
        order = rng.permutation(len(options))
        tasks += [make_task(name, options[order[i % len(options)]]) for i in range(episodes)]
    return tasks


def evaluate(env: ComputerUseEnv, policy: Policy, tasks: list[Task], split: str = "",
             verbose: bool = True) -> list[dict]:
    results = []
    for task in tasks:
        m = run_episode(env, policy, task)
        m["split"] = split
        results.append(m)
        if verbose:
            print(f"  {m['task']:12s} {m['text']!r:30s} {'ok  ' if m['success'] else 'FAIL'} "
                  f"t={m['time']:5.1f}s wrong_keys={m['wrong_keys']} failed_clicks={m['failed_clicks']} "
                  f"collisions={m['collisions']}")
    return results
