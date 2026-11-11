"""The fixed, versioned episode suite.

A split pairs a *content* split (which strings, queries, files the tasks use; see
environment/task_manager.SPLITS) with a *scene* distribution (environment/variation.py):

    train          training content    id scenes        for collecting demonstrations / RL
    val            training content    id scenes        model selection (separate seeds from train)
    test           held-out content    id scenes        main test: unseen strings, familiar scenes
    test_layout    held-out content    ood_layout       devices moved beyond the training range
    test_visual    held-out content    ood_visual       unseen lighting, colours, wallpaper, camera offset
    test_dynamics  held-out content    ood_dynamics     unseen cursor gain, key stiffness, mouse friction

Every episode is fully determined by its EpisodeSpec (task, text, scene distribution, seed),
and the specs of a split are a pure function of the benchmark version: seeds come from a
stable hash of (split, task, index), so two machines evaluate exactly the same episodes.
"""

from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass

import numpy as np

from environment.task_manager import SPLITS as CONTENT_SPLITS
from environment.task_manager import TASKS, Task, make_task
from environment.variation import SceneVariation, sample_variation


@dataclass(frozen=True)
class SplitSpec:
    content: str  # "train" | "test"
    scene: str  # variation distribution
    description: str


SPLITS: dict[str, SplitSpec] = {
    "train": SplitSpec("train", "id", "training content, in-distribution scenes"),
    "val": SplitSpec("train", "id", "training content, in-distribution scenes, own seeds: for model selection"),
    "test": SplitSpec("test", "id", "held-out content, in-distribution scenes"),
    "test_layout": SplitSpec("test", "ood_layout", "held-out content, keyboard and mouse moved beyond the training range"),
    "test_visual": SplitSpec("test", "ood_visual", "held-out content, unseen lighting / colours / wallpaper / camera offset"),
    "test_dynamics": SplitSpec("test", "ood_dynamics", "held-out content, unseen cursor gain / key stiffness / mouse friction"),
}
TEST_SPLITS = ("test", "test_layout", "test_visual", "test_dynamics")
TASK_ORDER = tuple(TASKS)  # sorted by level in task_manager
EPISODES_PER_TASK = 20  # per task and test split -> 180 episodes per split


def _seed(*parts: str | int) -> int:
    """Stable 32-bit seed (Python's hash() is salted per process)."""
    return zlib.crc32("/".join(map(str, parts)).encode())


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str  # "<split>/<task>/<index>"
    split: str
    task: str
    level: int
    text: str
    scene: str
    seed: int

    def make_task(self) -> Task:
        return make_task(self.task, self.text)

    def variation(self) -> SceneVariation:
        return sample_variation(self.scene, np.random.default_rng(self.seed))

    def to_dict(self) -> dict:
        return asdict(self)


def episodes(split: str, tasks: tuple[str, ...] | list[str] | None = None,
             per_task: int = EPISODES_PER_TASK) -> list[EpisodeSpec]:
    """The episode list of a split, in a fixed order."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {list(SPLITS)}")
    spec = SPLITS[split]
    out = []
    for task in tasks or TASK_ORDER:
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}")
        texts = CONTENT_SPLITS[task][spec.content]
        order = np.random.default_rng(_seed(split, task, "order")).permutation(len(texts))
        for i in range(per_task):
            out.append(EpisodeSpec(
                episode_id=f"{split}/{task}/{i:03d}",
                split=split,
                task=task,
                level=TASKS[task].level,
                text=texts[order[i % len(texts)]],
                scene=spec.scene,
                seed=_seed(split, task, i),
            ))
    return out
