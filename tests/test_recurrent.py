"""Recurrent behaviour cloning (learning/recurrent.py): inputs, training, and that the step-by-step
policy computes exactly what the network was trained to compute on whole episodes."""

import numpy as np
import pytest
import torch

from benchmark.agents import default_name, make_agent
from benchmark.harness import run_episode
from benchmark.suite import episodes
from benchmark.tracks import get_track
from environment.computer_env import ComputerUseEnv, EnvConfig
from learning.dataset import FEATURE_DIM, collect_demonstrations, decode_target
from learning.policy import BCPolicy, load_policy
from learning.recurrent import RecurrentBCNetwork, RecurrentBCPolicy, episode_inputs, train_recurrent_bc


@pytest.fixture(scope="module")
def env():
    e = ComputerUseEnv(EnvConfig(render_obs=False))
    yield e
    e.close()


@pytest.fixture(scope="module")
def demos(env):
    specs = episodes("train", tasks=["type_key", "open_app"], per_task=2)
    return collect_demonstrations(env, [s.make_task() for s in specs], seed=0, verbose=False,
                                  variations=[s.variation() for s in specs])


def test_inputs_carry_the_previous_action_within_each_episode(demos):
    eps = episode_inputs(demos)
    assert len(eps) == len(demos.task_specs)
    start = 0
    for x, y in eps:
        n = len(x)
        assert x.shape == (n, FEATURE_DIM + 4) and np.array_equal(y, demos.targets[start:start + n])
        assert np.array_equal(x[:, :FEATURE_DIM], demos.features[start:start + n])
        assert not x[0, FEATURE_DIM:].any()  # nothing before the episode starts
        assert np.array_equal(x[1:, FEATURE_DIM:], demos.actions[start:start + n - 1])
        start += n


def test_train_and_run(demos, env, tmp_path):
    ckpt = tmp_path / "gru.pt"
    history = train_recurrent_bc(demos, ckpt, epochs=2, batch_episodes=2, seed=0, device="cpu", verbose=False)
    assert len(history["train"]) == 2 and np.isfinite(history["val"]).all()
    assert isinstance(load_policy(ckpt, 0.006), RecurrentBCPolicy)
    assert default_name(f"bc:{ckpt}") == "bc-gru"
    spec = episodes("test", tasks=["type_key"], per_task=1)[0]
    rec = run_episode(env, make_agent(f"bc:{ckpt}", env), spec, get_track("state"))
    assert rec["steps"] > 0


def test_mlp_checkpoints_still_load_as_bc(tmp_path):
    from learning.policy import BCNetwork

    ckpt = tmp_path / "mlp.pt"
    torch.save({"state_dict": BCNetwork().state_dict(), "feature_dim": FEATURE_DIM, "image_keys": ()}, ckpt)
    assert isinstance(load_policy(ckpt, 0.006), BCPolicy) and default_name(f"bc:{ckpt}") == "bc"


def test_stepwise_policy_matches_the_unrolled_network(env, tmp_path):
    torch.manual_seed(0)
    ckpt = tmp_path / "gru.pt"
    net = RecurrentBCNetwork()
    torch.save({"kind": "recurrent_bc", "state_dict": net.state_dict(), "feature_dim": FEATURE_DIM, "hidden": 256},
               ckpt)
    policy = RecurrentBCPolicy(ckpt, env.config.max_step)
    spec = episodes("test", tasks=["open_app"], per_task=1)[0]
    obs = env.reset(spec.make_task(), variation=spec.variation())
    policy.reset()
    from learning.dataset import low_dim_features, normalize_action

    feats, actions = [], []
    for _ in range(30):
        feats.append(low_dim_features(obs))
        actions.append(policy.act(obs))
        obs, *_ = env.step(actions[-1])
    prev = np.zeros((30, 4), np.float32)
    prev[1:] = [normalize_action(a, env.config.max_step) for a in actions[:-1]]
    x = torch.from_numpy(np.concatenate([np.array(feats), prev], axis=1)).unsqueeze(0)
    with torch.no_grad():
        pred, _ = net(x)
    unrolled = [decode_target(p, env.config.max_step) for p in pred[0].numpy()]
    np.testing.assert_allclose(np.array(actions), np.array(unrolled), atol=1e-6)
    # reset() forgets the episode: the first action depends on the first observation only.
    policy.reset()
    obs = env.reset(spec.make_task(), variation=spec.variation())
    np.testing.assert_allclose(policy.act(obs), actions[0], atol=1e-6)


def test_dagger_student_can_be_recurrent(env, demos, tmp_path):
    from benchmark import harness

    ckpt = tmp_path / "gru.pt"
    train_recurrent_bc(demos, ckpt, epochs=1, batch_episodes=2, seed=0, device="cpu", verbose=False)
    harness._init_collector("nominal", str(ckpt))
    assert isinstance(harness._COLLECT["student"], RecurrentBCPolicy)
    harness._COLLECT.pop("env").close()
