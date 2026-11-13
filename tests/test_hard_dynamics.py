"""The realistic physics profile and its hidden dynamics: each effect does what it models,
the hard splits are valid (solvable, OOD, deterministic) and nominal physics is untouched."""

import dataclasses

import mujoco
import numpy as np
import pytest

from benchmark.agents import make_agent
from benchmark.harness import run_episode
from benchmark.suite import HARD_TEST_SPLITS, SPLITS, episodes
from benchmark.tracks import get_track
from control.contact_detection import DeviceMonitor
from environment.browser import Desktop
from environment.computer_env import ComputerUseEnv, EnvConfig, build_model
from environment.dynamics import NOMINAL_DYNAMICS, sample_dynamics
from environment.keyboard_layout import KEYS
from environment.task_manager import make_task
from environment.ui_events import CURSOR_GAIN, PhysicalUIBridge
from learning.dataset import collect_demonstrations


@pytest.fixture(scope="module")
def real_env():
    e = ComputerUseEnv(EnvConfig(render_obs=False, physics="realistic"))
    yield e
    e.close()


def test_realistic_model_adds_only_the_mouse_rotation():
    nominal, realistic = build_model("nominal"), build_model("realistic")
    assert realistic.nq == nominal.nq + 1 and realistic.joint("mouse_yaw").type == mujoco.mjtJoint.mjJNT_HINGE
    with pytest.raises(ValueError):
        ComputerUseEnv(EnvConfig(render_obs=False)).reset(make_task("type_key", "a"),
                                                          dynamics=sample_dynamics("real_id", np.random.default_rng(0),
                                                                                   len(KEYS)))


def test_rotated_mouse_moves_cursor_in_its_own_frame():
    model = build_model("realistic")
    data = mujoco.MjData(model)
    desktop = Desktop()
    bridge = PhysicalUIBridge(DeviceMonitor(model), desktop, yaw_qadr=model.joint("mouse_yaw").qposadr[0])
    # Straight to the robot's right (-y): pure +u for an unrotated mouse ...
    du, dv = bridge.cursor_delta(np.array([0.0, -0.01]), yaw=0.0, speed=0.0)
    assert du == pytest.approx(CURSOR_GAIN * 0.01) and dv == pytest.approx(0.0)
    # ... but diagonal for a mouse rotated by 30 degrees.
    du, dv = bridge.cursor_delta(np.array([0.0, -0.01]), yaw=np.deg2rad(30), speed=0.0)
    assert du == pytest.approx(CURSOR_GAIN * 0.01 * np.cos(np.deg2rad(30)))
    assert abs(dv) == pytest.approx(CURSOR_GAIN * 0.01 * np.sin(np.deg2rad(30)))
    bridge.pointer_accel = 1.0
    fast, _ = bridge.cursor_delta(np.array([0.0, -0.01]), yaw=0.0, speed=0.3)
    assert fast == pytest.approx(3 * CURSOR_GAIN * 0.01)  # acceleration triples the gain at speed


def test_held_key_autorepeats():
    model = build_model("realistic")
    data = mujoco.MjData(model)
    desktop = Desktop()
    desktop.reset(active="editor")
    devices = DeviceMonitor(model)
    devices.autorepeat_delay = 0.3
    bridge = PhysicalUIBridge(devices, desktop)
    bridge.reset(data)
    data.qpos[model.joint("key_k").qposadr[0]] = -0.0035
    for t in np.arange(0.0, 0.5, 0.002):  # hold "k" for half a second
        data.time = t
        bridge.update(data)
    # one press, then repeats at 30 Hz from 0.3 s on
    assert desktop.apps["editor"].text == "k" * (1 + int((0.5 - 0.3) * 30 + 0.999))


def test_files_need_a_double_click_within_the_window():
    d = Desktop()
    d.reset(active="files")
    d.double_click_time = 0.4
    d.set_cursor(d.widget_rect("files:item:2").center)
    d.click(time=1.0)
    assert d.active == "files" and d.apps["files"].selected == 2  # single click selects
    d.click(time=1.6)  # too slow: still just a selection
    assert d.active == "files"
    d.click(time=1.9)
    assert d.active == "editor" and d.apps["editor"].filename == "papers.txt"


def test_control_latency_delays_actions(real_env):
    dyn = dataclasses.replace(NOMINAL_DYNAMICS, action_delay=3, name="probe")
    real_env.reset(make_task("type_key", "a"), dynamics=dyn)
    start = real_env.tool_target.copy()
    real_env.step(np.array([0.006, 0, 0, 0]))
    real_env.step(np.zeros(4))
    real_env.step(np.zeros(4))
    assert np.allclose(real_env.tool_target, start)  # not applied yet
    assert real_env.planned_tool_target()[0] == pytest.approx(start[0] + 0.006)
    real_env.step(np.zeros(4))
    assert real_env.tool_target[0] == pytest.approx(start[0] + 0.006)


def test_display_latency_lags_the_observed_cursor(real_env):
    dyn = dataclasses.replace(NOMINAL_DYNAMICS, display_latency=3, name="probe")
    obs = real_env.reset(make_task("type_key", "a"), dynamics=dyn)
    start = tuple(obs["cursor"])
    real_env.desktop.set_cursor((100, 100))
    lagged = [tuple(real_env.step(np.zeros(4))[0]["cursor"]) for _ in range(4)]
    assert lagged[:3] == [start] * 3 and lagged[3] == (100, 100)


def test_hard_splits_are_deterministic_and_out_of_distribution():
    for split in ("hard_train", "hard_val", "hard_test", "hard_test_ood"):
        a = episodes(split, per_task=2)
        assert a == episodes(split, per_task=2) and all(e.physics == "realistic" for e in a)
        assert a[0].hidden_dynamics() == a[0].hidden_dynamics()
    rng = np.random.default_rng(0)
    id_ = [sample_dynamics("real_id", rng, len(KEYS)) for _ in range(200)]
    ood = [sample_dynamics("real_ood", rng, len(KEYS)) for _ in range(200)]
    for field, hi_id, lo_ood in (("action_delay", 1, 2), ("display_latency", 2, 3), ("pointer_accel", 0.5, 0.8),
                                 ("action_noise", 0.0002, 0.0003), ("obs_noise", 0.0003, 0.0005)):
        assert max(getattr(d, field) for d in id_) <= hi_id < lo_ood <= min(getattr(d, field) for d in ood), field
    assert min(d.gravcomp for d in id_) >= 0.95 > 0.9 >= max(d.gravcomp for d in ood)
    assert max(abs(d.mouse_yaw) for d in id_) <= 0.15 < 0.25 <= min(abs(d.mouse_yaw) for d in ood)
    assert min(d.autorepeat_delay for d in id_) >= 0.45 > 0.35 >= max(d.autorepeat_delay for d in ood)
    assert min(min(d.key_stiffness) for d in ood) >= 1.5 > 1.4 >= max(max(d.key_stiffness) for d in id_)
    assert SPLITS["test"].physics == "nominal" and "dynamics" not in episodes("test", per_task=1)[0].to_dict()


def test_dagger_executes_the_student_and_keeps_oracle_labels(real_env):
    class Frozen:  # a student that never moves
        def reset(self):
            self.calls = 0

        def act(self, obs):
            self.calls += 1
            return np.zeros(4)

    student = Frozen()
    demos = collect_demonstrations(real_env, [make_task("type_key", "a")], verbose=False, keep_failures=True,
                                   student=student)
    assert not real_env.status.success and len(demos.task_specs) == 1  # failed episode kept
    assert len(demos) == student.calls == real_env.metrics()["steps"]
    assert np.allclose(demos.actions, 0.0)  # the student's actions were executed ...
    assert np.linalg.norm(demos.targets[-1, :3]) > 1.0  # ... and the oracle still points to the key (cm)


def test_dynamics_ablation_changes_exactly_one_factor(real_env):
    from benchmark.ablation import ALL_FIELDS, CONDITIONS, FACTORS, condition_dynamics

    spec = episodes("hard_test", per_task=1)[0]
    base = spec.hidden_dynamics()
    assert condition_dynamics(spec, "in range") == base
    for condition, fields in FACTORS.items():
        d = condition_dynamics(spec, condition)
        changed = {f for f in ALL_FIELDS if getattr(d, f) != getattr(base, f)}
        assert changed == set(fields), condition
    assert {f for f in ALL_FIELDS if getattr(condition_dynamics(spec, "all"), f) != getattr(base, f)} == set(ALL_FIELDS)
    # the oracle reads the hidden parameters, so it still solves every condition
    agent = make_agent("oracle", real_env)
    for condition in ("control latency", "mouse rotation", "double-click time", "all"):
        for spec in episodes("hard_test", tasks=["open_file"], per_task=1):
            rec = run_episode(real_env, agent, spec, get_track("state"), dynamics=condition_dynamics(spec, condition))
            assert rec["success"], condition
    assert len(CONDITIONS) == len(FACTORS) + 2


@pytest.mark.parametrize("split", HARD_TEST_SPLITS)
def test_oracle_solves_every_task_under_hidden_dynamics(real_env, split):
    agent = make_agent("oracle", real_env)
    for spec in episodes(split, per_task=1):
        rec = run_episode(real_env, agent, spec, get_track("state"))
        assert rec["success"], spec
