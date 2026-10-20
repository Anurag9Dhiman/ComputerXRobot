"""Milestone 1: the robot presses a physical key, the character appears in the simulated
text editor, and the task manager verifies it."""

import numpy as np

from environment.task_manager import make_task
from evaluation.task_success import run_episode
from learning.scripted_policy import ScriptedPolicy


def test_robot_presses_key_and_editor_shows_character(env):
    metrics = run_episode(env, ScriptedPolicy(env), make_task("type_key", "a"))

    assert metrics["success"]
    assert env.desktop.apps["editor"].text == "a"
    keys = [e for e in env.tasks.events if e.kind == "key"]
    assert [(e.value, e.correct) for e in keys] == [("a", True)]
    assert metrics["wrong_keys"] == 0
    assert metrics["collisions"] == 0


def test_key_press_comes_from_physics(env):
    """The character is produced by the key's joint travelling, i.e. by contact."""
    env.reset(make_task("type_key", "q"))
    joint = env.model.joint("key_q")
    q = env.data.qpos[joint.qposadr[0]]
    assert q > -0.001  # key at rest
    policy = ScriptedPolicy(env)
    min_q = q
    while not env.status.done:
        env.step(policy.act())
        min_q = min(min_q, env.data.qpos[joint.qposadr[0]])
    assert env.status.success
    assert min_q < -0.0024  # travelled past the 60% actuation point of the 4 mm stroke


def test_idle_robot_types_nothing(env):
    env.reset(make_task("type_key", "a"))
    for _ in range(50):
        env.step(np.zeros(4))
    assert env.desktop.apps["editor"].text == ""
    assert not env.status.success
