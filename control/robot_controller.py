"""Inverse kinematics and low-level control for the 5-DOF arm.

The policy commands a Cartesian target for the `tool` site plus a finger extension. The
controller converts that into joint position targets with damped-least-squares IK (holding
the hand pointing straight down with a fixed heading) and writes them to the position
actuators. Gravity is compensated in the model (body `gravcomp`), so the position servos
track accurately.
"""

from __future__ import annotations

import mujoco
import numpy as np

ARM_JOINTS = ("yaw", "shoulder", "elbow", "wrist_pitch", "wrist_roll")
FINGER_JOINT = "finger"
TOOL_SITE = "tool"

# Joint configuration used to seed IK: elbow up, hand pointing down.
IK_SEED = np.deg2rad([0.0, -45.0, 100.0, 35.0, 0.0])


class RobotController:
    def __init__(
        self,
        model: mujoco.MjModel,
        rot_weight: float = 0.5,
        damping: float = 0.02,
        ik_iters_per_step: int = 10,
    ):
        self.model = model
        self.rot_weight = rot_weight
        self.damping = damping
        self.ik_iters_per_step = ik_iters_per_step
        self._ik_data = mujoco.MjData(model)

        joints = [model.joint(name) for name in ARM_JOINTS]
        self.arm_qadr = np.array([j.qposadr[0] for j in joints])
        self.arm_dadr = np.array([j.dofadr[0] for j in joints])
        self.arm_range = np.array([model.jnt_range[j.id] for j in joints])
        self.arm_act = np.array([model.actuator(name).id for name in ARM_JOINTS])

        finger = model.joint(FINGER_JOINT)
        self.finger_qadr = finger.qposadr[0]
        self.finger_max = float(model.jnt_range[finger.id][1])
        self.finger_act = model.actuator(FINGER_JOINT).id

        self.site_id = model.site(TOOL_SITE).id
        # The tool frame is world-aligned when the hand points straight down.
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.q_des = IK_SEED.copy()

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # ------------------------------------------------------------------ IK
    def solve_ik(
        self, target_pos: np.ndarray, q_init: np.ndarray, max_iters: int = 100, tol: float = 1e-6
    ) -> tuple[np.ndarray, float]:
        """Damped least squares IK for the tool site. Returns (q, final error norm)."""
        m, d = self.model, self._ik_data
        q = q_init.copy()
        cur_quat, neg_quat, err_quat = np.zeros(4), np.zeros(4), np.zeros(4)
        err_rot = np.zeros(3)
        err_norm = np.inf
        for _ in range(max_iters):
            d.qpos[self.arm_qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)

            err_pos = target_pos - d.site_xpos[self.site_id]
            mujoco.mju_mat2Quat(cur_quat, d.site_xmat[self.site_id])
            mujoco.mju_negQuat(neg_quat, cur_quat)
            mujoco.mju_mulQuat(err_quat, self.target_quat, neg_quat)
            if err_quat[0] < 0:
                err_quat *= -1
            mujoco.mju_quat2Vel(err_rot, err_quat, 1.0)
            err = np.concatenate([err_pos, self.rot_weight * err_rot])
            err_norm = float(np.linalg.norm(err))
            if err_norm < tol:
                break

            mujoco.mj_jacSite(m, d, self._jacp, self._jacr, self.site_id)
            jac = np.vstack(
                [self._jacp[:, self.arm_dadr], self.rot_weight * self._jacr[:, self.arm_dadr]]
            )
            # Limit the step so a far target does not throw the arm into another IK branch.
            step = err * min(1.0, 0.05 / max(err_norm, 1e-9))
            dq = jac.T @ np.linalg.solve(jac @ jac.T + self.damping**2 * np.eye(6), step)
            q = np.clip(q + dq, self.arm_range[:, 0], self.arm_range[:, 1])
        return q, err_norm

    # --------------------------------------------------------- control API
    def reset(self, data: mujoco.MjData, tool_pos: np.ndarray) -> None:
        """Teleport the arm to the IK solution for `tool_pos` (finger retracted)."""
        q, err = self.solve_ik(np.asarray(tool_pos, float), IK_SEED, max_iters=300)
        if err > 1e-3:
            raise ValueError(f"home pose {tool_pos} is not reachable (IK error {err:.4f})")
        self.q_des = q
        data.qpos[self.arm_qadr] = q
        data.qvel[self.arm_dadr] = 0.0
        data.qpos[self.finger_qadr] = 0.0
        data.ctrl[self.arm_act] = q
        data.ctrl[self.finger_act] = 0.0

    def apply(self, data: mujoco.MjData, tool_target: np.ndarray, finger_cmd: float) -> None:
        """Set actuator targets for a tool-site target and a finger command in [0, 1]."""
        self.q_des, _ = self.solve_ik(tool_target, self.q_des, max_iters=self.ik_iters_per_step)
        data.ctrl[self.arm_act] = self.q_des
        data.ctrl[self.finger_act] = float(np.clip(finger_cmd, 0.0, 1.0)) * self.finger_max

    # ------------------------------------------------------------- state
    def tool_pos(self, data: mujoco.MjData) -> np.ndarray:
        return data.site_xpos[self.site_id].copy()

    def finger_extension(self, data: mujoco.MjData) -> float:
        return float(data.qpos[self.finger_qadr])

    def arm_qpos(self, data: mujoco.MjData) -> np.ndarray:
        return data.qpos[self.arm_qadr].copy()

    def arm_qvel(self, data: mujoco.MjData) -> np.ndarray:
        return data.qvel[self.arm_dadr].copy()
