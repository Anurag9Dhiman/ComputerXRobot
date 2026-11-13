"""Validity of the benchmark itself: determinism, split design, solvability, track isolation,
harness reproducibility and scoring."""

import numpy as np
import pytest

from benchmark.harness import evaluate, run_episode
from benchmark.agents import make_agent
from benchmark.metrics import summarize
from benchmark.report import report
from benchmark.suite import EPISODES_PER_TASK, SPLITS, TASK_ORDER, TEST_SPLITS, episodes
from benchmark.tracks import TRACKS, get_track
from environment.task_manager import SPLITS as CONTENT
from environment.task_manager import TASKS, make_task
from environment.variation import (ID_DESKS, ID_WALLPAPERS, NOMINAL, sample_variation)


def test_episode_lists_are_fixed_and_cover_every_task():
    for split in SPLITS:
        a, b = episodes(split), episodes(split)
        assert a == b
        assert len(a) == EPISODES_PER_TASK * len(TASKS)
        assert len({e.episode_id for e in a}) == len(a)
        assert {e.task for e in a} == set(TASKS)
        content = SPLITS[split].content
        assert all(e.text in CONTENT[e.task][content] for e in a)
        assert all(e.scene == SPLITS[split].scene for e in a)
    assert episodes("test")[0].seed != episodes("test_layout")[0].seed


def test_ood_scene_ranges_are_disjoint_from_training():
    rng = np.random.default_rng(0)
    id_ = [sample_variation("id", rng) for _ in range(300)]
    lay = [sample_variation("ood_layout", rng) for _ in range(300)]
    vis = [sample_variation("ood_visual", rng) for _ in range(300)]
    dyn = [sample_variation("ood_dynamics", rng) for _ in range(300)]

    assert max(abs(v.keyboard_offset[0]) for v in id_) <= 0.02 < 0.03 <= min(abs(v.keyboard_offset[0]) for v in lay)
    assert max(v.keyboard_offset[1] for v in id_) <= 0.03 < 0.04 <= min(v.keyboard_offset[1] for v in lay)
    assert min(v.mouse_offset[1] for v in id_) >= -0.03 > -0.04 >= max(v.mouse_offset[1] for v in lay)

    assert all(not 0.8 <= v.light_scale <= 1.2 for v in vis)
    assert all(abs(v.light_azimuth) >= 0.6 for v in vis) and all(abs(v.light_azimuth) <= 0.3 for v in id_)
    assert not {v.desk_rgb for v in vis} & set(ID_DESKS)
    assert not {v.wallpaper_rgb for v in vis} & set(ID_WALLPAPERS)
    assert all(min(map(abs, v.camera_offset)) >= 0.015 for v in vis)

    assert all(not 5600 <= v.cursor_gain <= 7200 for v in dyn)
    assert all(v.key_stiffness_scale >= 1.3 and v.mouse_friction_scale >= 1.6 for v in dyn)
    assert all(v.key_stiffness_scale <= 1.1 and v.mouse_friction_scale <= 1.2 for v in id_)


def test_variations_apply_and_never_accumulate(env):
    kb = env.model.body("keyboard").id
    nominal = env.model.body_pos[kb].copy()
    var = sample_variation("ood_layout", np.random.default_rng(1))
    env.reset(make_task("type_key", "a"), variation=var)
    assert np.allclose(env.model.body_pos[kb][:2], nominal[:2] + var.keyboard_offset)
    assert env.bridge.gain == var.cursor_gain
    env.reset(make_task("type_key", "a"), variation=var)
    env.reset(make_task("type_key", "a"), variation=NOMINAL)
    assert np.array_equal(env.model.body_pos[kb], nominal)
    assert np.allclose(env.key_top("a")[:2], env.data.xpos[env.model.body("key_a").id][:2])


@pytest.mark.parametrize("split", TEST_SPLITS)
def test_oracle_solves_every_task(env, split):
    agent = make_agent("oracle", env)
    for spec in episodes(split, per_task=1):
        rec = run_episode(env, agent, spec, get_track("state"))
        assert rec["success"], spec
        assert rec["wrong_keys"] == rec["failed_clicks"] == rec["collisions"] == 0, spec
    env.reset(make_task("type_key", "a"), variation=NOMINAL)


def test_vision_track_hides_privileged_state():
    vision = get_track("vision")
    hidden = {"expected_char", "click_target", "click_target_px", "click_button", "mouse_pos", "mouse_button",
              "cursor", "keyboard_pos", "key_down", "button_down", "screen"}
    assert not hidden & set(vision.keys)
    assert set(vision.cameras) <= set(vision.keys) and "instruction" in vision.keys
    obs = {k: 0 for k in TRACKS["state"].keys + vision.keys + ("screen",)}
    assert set(vision.filter(obs)) == set(vision.keys)


def test_parallel_evaluation_matches_sequential():
    kwargs = dict(tasks=["type_key", "close_window"], per_task=2, verbose=False)
    seq = evaluate("random", "test_dynamics", workers=1, **kwargs)
    par = evaluate("random", "test_dynamics", workers=2, **kwargs)
    assert seq["episodes"] == par["episodes"]
    assert seq["summary"]["score"] == par["summary"]["score"]
    assert seq["meta"]["version"] and seq["track"] == "state"


def _record(task, success):
    return {"task": task, "level": TASKS[task].level, "success": success, "time": 1.0, "return": 0.0,
            "wrong_keys": 0, "failed_clicks": 0, "collisions": 0, "failed_contacts": 0}


def test_score_is_task_macro_average_with_ci():
    records = [_record("type_key", True)] * 9 + [_record("type_key", False)] + [_record("search", False)] * 30
    s = summarize(records)
    assert s["score"] == pytest.approx((0.9 + 0.0) / 2)  # tasks weigh equally, not episodes
    assert s["score_ci"][0] <= s["score"] <= s["score_ci"][1]
    assert s["by_task"]["type_key"]["ci"][1] <= 1.0 and s["by_level"] == {"1": 0.9, "3": 0.0}


def test_report_tables_render():
    results = []
    for agent, rate in (("oracle", True), ("random", False)):
        for split in TEST_SPLITS:
            records = [{**_record(t, rate), "episode_id": f"{split}/{t}/{i:03d}"} for t in TASK_ORDER for i in range(3)]
            results.append({"agent_name": agent, "privileged": agent == "oracle", "track": "state", "split": split,
                            "summary": summarize(records), "episodes": records})
    md = report(results, "splits", fmt="md")
    assert "oracle (privileged)" in md and "| random | state | 0.0 [0.0, 0.0]" in md
    assert "| random | state | 0.0 ± 0.0" in report(results, "splits", interval="std")
    assert "hard" not in report(results, "hard").split("\n")[2]  # no hard results: header only
    assert r"\begin{tabular}" in report(results, "tasks", "test", fmt="latex")
    for table in ("levels", "errors"):
        assert report(results, table, "test_visual")


def test_plugin_agent_and_vision_observations():
    from benchmark.agents import agent_class
    from environment.computer_env import ComputerUseEnv

    cls, _ = agent_class("benchmark.example_agent:ExampleVisionAgent")
    assert cls.track == "vision" and cls.name == "example-vision"
    track = get_track("vision")
    env = ComputerUseEnv(track.env_config())
    spec = episodes("test_visual", ["type_key"], per_task=1)[0]
    obs = track.filter(env.reset(spec.make_task(), variation=spec.variation()))
    assert set(obs) == set(track.keys)
    assert obs["keyboard_cam"].shape == (256, 512, 3) and obs["screen_cam"].shape == (400, 640, 3)
    assert obs["instruction"] == spec.make_task().instruction
    action = cls(env).act(obs)
    assert action.shape == (4,)
    env.close()
