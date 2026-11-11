"""The non-privileged adaptive expert (learning/adaptive.py): it never reads the simulator,
identifies the hidden latencies from its observations, and that identification is what
solves the beyond-range dynamics (the same law without it fails)."""

import numpy as np
import pytest

from benchmark.agents import make_agent
from benchmark.harness import run_episode
from benchmark.suite import episodes
from benchmark.tracks import get_track
from environment.computer_env import ComputerUseEnv, EnvConfig
from learning.adaptive import ObservedWorld
from learning.scripted_policy import ScriptedPolicy, SimWorld


@pytest.fixture(scope="module")
def real_env():
    e = ComputerUseEnv(EnvConfig(render_obs=False, physics="realistic"))
    yield e
    e.close()


def _references(obj, seen=None):
    """Every object reachable from obj's attributes (to prove no path leads to the env)."""
    seen = set() if seen is None else seen
    if id(obj) in seen or isinstance(obj, (int, float, str, bool, type(None), np.ndarray)):
        return seen
    seen.add(id(obj))
    children = obj.values() if isinstance(obj, dict) else obj if isinstance(obj, (list, tuple, set)) else \
        vars(obj).values() if hasattr(obj, "__dict__") else ()
    for child in children:
        _references(child, seen)
    return seen


def test_adaptive_expert_holds_no_reference_to_the_simulator(real_env):
    agent = make_agent("adaptive", real_env)
    assert not agent.privileged and agent.track == "state"
    reachable = _references(agent)
    for sim_object in (real_env, real_env.data, real_env.model, real_env.desktop, real_env.tasks, real_env.devices):
        assert id(sim_object) not in reachable
    assert isinstance(agent.policy.world, ObservedWorld) and agent.policy.policy.env is None


def test_oracle_view_is_the_simulator(real_env):
    assert isinstance(ScriptedPolicy(real_env).world, SimWorld)
    with pytest.raises(ValueError):
        ScriptedPolicy()


@pytest.mark.parametrize("task", ["type_key", "open_app", "close_window"])
def test_identification_solves_beyond_range_dynamics(real_env, task):
    spec = episodes("hard_test_ood", tasks=[task], per_task=1)[0]
    dyn = spec.hidden_dynamics()
    assert dyn.action_delay >= 2 and dyn.display_latency >= 3  # beyond the training range
    adaptive = make_agent("adaptive", real_env)
    assert run_episode(real_env, adaptive, spec, get_track("state"))["success"]
    world = adaptive.policy.world
    assert world.control_delay == dyn.action_delay
    if task != "type_key":  # typing never moves the mouse, so there is no cursor evidence
        assert world.lag_identified and world.display_lag == dyn.display_latency
    # The same control law assuming nominal dynamics fails on the same episode.
    assert not run_episode(real_env, make_agent("fixed", real_env), spec, get_track("state"))["success"]
