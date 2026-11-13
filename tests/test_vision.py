"""Vision-track behaviour cloning (learning/vision.py) and the vision_subgoal diagnostic track."""

import numpy as np
import pytest
import torch

from benchmark.agents import default_name
from benchmark.suite import episodes
from benchmark.tracks import get_track
from environment.computer_env import ComputerUseEnv
from learning.vision import (CAMERAS, PROPRIO_DIM, SUBGOAL_DIM, VisionBCNetwork, VisionBCPolicy, downsample,
                             encode_instruction, vision_inputs)


@pytest.fixture(scope="module")
def vision_obs():
    env = ComputerUseEnv(get_track("vision").env_config())
    spec = episodes("test", tasks=["open_app"], per_task=1)[0]
    obs = env.reset(spec.make_task(), variation=spec.variation())
    yield obs
    env.close()


def test_downsample_is_a_rounded_box_filter():
    img = np.random.default_rng(0).integers(0, 256, (10, 15, 3), dtype=np.uint8)
    out = downsample(img, 5)
    expect = np.floor(img.reshape(2, 5, 3, 5, 3).mean((1, 3)) + 0.5).astype(np.uint8)
    assert out.shape == (2, 3, 3) and np.array_equal(out, expect)


def test_inputs_use_only_the_vision_track(vision_obs):
    seen = get_track("vision").filter(vision_obs)
    x = vision_inputs({**seen, "expected_char": None, "click_target_px": None})
    sizes = {"robot_cam": (48, 64), "keyboard_cam": (64, 128), "mouse_cam": (64, 64), "screen_cam": (80, 128)}
    assert set(sizes) == set(CAMERAS)
    for cam, hw in sizes.items():
        assert x[cam].shape == (*hw, 3) and x[cam].dtype == np.uint8
    assert x["proprio"].shape == (PROPRIO_DIM,) and x["subgoal"].shape == (SUBGOAL_DIM,)
    assert not x["subgoal"].any()  # nothing leaks from the state track
    assert encode_instruction(seen["instruction"])[0] == ord(seen["instruction"][0]) - 31


def test_collector_records_vision_inputs_every_second_step():
    from learning.dataset import collect_demonstrations

    env = ComputerUseEnv(get_track("vision").env_config())
    spec = episodes("train", tasks=["type_key"], per_task=1)[0]
    kw = dict(noise=0.5, seed=0, verbose=False, variations=[spec.variation()])
    every = collect_demonstrations(env, [spec.make_task()], **kw)
    half = collect_demonstrations(env, [spec.make_task()], observe=vision_inputs, record_every=2, **kw)
    env.close()
    # Same rollout (same perturbations), every second step recorded, dtypes kept.
    assert np.array_equal(half.targets, every.targets[::2]) and np.array_equal(half.actions, every.actions[::2])
    assert half.images["keyboard_cam"].dtype == np.uint8 and half.images["proprio"].dtype == np.float32
    assert len(half.images["instruction"]) == len(half) and half.images["subgoal"][:, :-4].any()


def test_subgoal_track_adds_only_the_subgoal():
    vision, diag = get_track("vision"), get_track("vision_subgoal")
    assert set(diag.keys) - set(vision.keys) == {"expected_char", "click_target", "click_button", "click_target_px"}
    assert diag.cameras == vision.cameras


@pytest.mark.parametrize("use_subgoal", [False, True])
def test_policy_acts_on_its_track(vision_obs, tmp_path, use_subgoal):
    torch.manual_seed(0)
    ckpt = tmp_path / "vbc.pt"
    torch.save({"kind": "vision_bc", "state_dict": VisionBCNetwork(use_subgoal).state_dict(), "use_subgoal": use_subgoal,
                "cameras": CAMERAS}, ckpt)
    policy = VisionBCPolicy(ckpt, max_step=0.006)
    track = get_track("vision_subgoal" if use_subgoal else "vision")
    action = policy.act(track.filter(vision_obs))
    assert action.shape == (4,) and np.all(np.abs(action[:3]) <= 0.006 + 1e-9) and 0 <= action[3] <= 1
    assert default_name(f"vbc:{ckpt}") == ("vision-bc-subgoal" if use_subgoal else "vision-bc")
    if use_subgoal:
        with pytest.raises(KeyError):
            policy.act(get_track("vision").filter(vision_obs))
