"""Robot adapter for the named Menagerie Panda actuators and state."""

from contextlib import nullcontext

import mujoco
import numpy as np

from recovera.environment import MujocoEnvironment


class PandaAdapter:
    """Map Panda's seven position servos, tendon gripper and Cartesian grasp site."""

    ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
    ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))

    def __init__(self, env: MujocoEnvironment):
        self.env, self.model, self.data = env, env.model, env.data
        self.joints = [self._id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.ARM_JOINTS]
        self.actuators = [self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.ARM_ACTUATORS]
        self.gripper = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.site = self._id(mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        self.cube_geom = self._id(mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.left_finger = self._id(mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger = self._id(mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self.finger_qpos = [
            int(self.model.jnt_qposadr[self._id(mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in ("finger_joint1", "finger_joint2")
        ]
        self.qpos = self.model.jnt_qposadr[self.joints]
        self.dofs = self.model.jnt_dofadr[self.joints]
        self.ranges = self.model.actuator_ctrlrange[self.actuators].copy()
        for joint_id, actuator_id, name in zip(self.joints, self.actuators, self.ARM_ACTUATORS):
            if self.model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError(f"Unsupported Panda model: {name} is not a joint actuator")
            if self.model.actuator_trnid[actuator_id, 0] != joint_id:
                raise ValueError(
                    f"Unsupported Panda model: {name} does not control its named joint"
                )

    def _id(self, kind: mujoco.mjtObj, name: str) -> int:
        value = mujoco.mj_name2id(self.model, kind, name)
        if value < 0:
            raise ValueError(f"Unsupported Panda model: required {kind.name} {name!r} is missing")
        return value

    @property
    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.qpos].copy()

    @property
    def end_effector_position(self) -> np.ndarray:
        return self.data.site_xpos[self.site].copy()

    @property
    def cube_position(self) -> np.ndarray:
        return self.env.body_position("cube")

    def command_gripper(self, opening: float) -> None:
        """Set approximate finger gap in metres (0 closed, 0.08 open)."""
        if not 0 <= opening <= 0.08:
            raise ValueError("gripper opening must be within [0, 0.08] metres")
        high = self.model.actuator_ctrlrange[self.gripper, 1]
        with self._viewer_lock():
            self.data.ctrl[self.gripper] = opening / 0.08 * high

    def initialize(self) -> bool:
        """Confirm named home state and open the coupled gripper before task motion."""
        home_qpos = self.model.key_qpos[self.env.home_keyframe]
        expected = home_qpos[self.qpos]
        if not np.allclose(self.joint_positions, expected, atol=1e-6):
            raise RuntimeError("Panda did not reset to the model's named home configuration")
        self.command_gripper(0.08)
        self.env.step(250)
        if self.env.viewer_closed:
            return False
        measured_opening = float(sum(self.data.qpos[address] for address in self.finger_qpos))
        return abs(measured_opening - 0.08) <= 0.004

    def hold_gripper(self, preload: float = 0.020) -> None:
        """Hold inside the measured bilateral contact aperture to maintain grip force."""
        opening = float(sum(self.data.qpos[address] for address in self.finger_qpos))
        self.command_gripper(max(0.0, opening - preload))

    def has_grasp_contact(self) -> bool:
        """Return true only when the cube touches both Panda finger bodies."""
        touched: set[int] = set()
        for contact in self.data.contact[: self.data.ncon]:
            if self.cube_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.cube_geom else contact.geom1
            body = int(self.model.geom_bodyid[other])
            if body in (self.left_finger, self.right_finger):
                touched.add(body)
        return touched == {self.left_finger, self.right_finger}

    def move_to(
        self,
        target: np.ndarray,
        max_steps: int,
        tolerance: float,
        target_rotation: np.ndarray | None = None,
        max_joint_step: float = 0.035,
        maintain_grasp: bool = False,
    ) -> bool:
        """Damped least-squares IK tracks site position and optional world rotation."""
        tol = 1.5 * tolerance
        for step_index in range(max_steps):
            position_error = np.asarray(target) - self.end_effector_position
            rotation_error = self._rotation_error(target_rotation)
            position_ok = np.linalg.norm(position_error) <= tol
            rotation_ok = rotation_error is None or np.linalg.norm(rotation_error) <= 0.05
            if position_ok and rotation_ok and step_index >= 50:
                self._hold_current_position()
                return True
            self.track_pose_step(target, target_rotation, max_joint_step)
            self.env.step()
            if self.env.viewer_closed:
                return False
        position_ok = (
            np.linalg.norm(np.asarray(target) - self.end_effector_position) <= 1.5 * tolerance
        )
        rotation_error = self._rotation_error(target_rotation)
        rotation_ok = rotation_error is None or np.linalg.norm(rotation_error) <= 0.06
        converged = position_ok and rotation_ok
        if converged:
            self._hold_current_position()
        return bool(converged)

    def track_pose_step(
        self,
        target: np.ndarray,
        target_rotation: np.ndarray | None = None,
        max_joint_step: float = 0.035,
    ) -> None:
        """Apply one Cartesian feedback update without advancing simulation time."""
        jac_position = np.zeros((3, self.model.nv))
        jac_rotation = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_position, jac_rotation, self.site)
        position_error = np.asarray(target) - self.end_effector_position
        rotation_error = self._rotation_error(target_rotation)
        if rotation_error is None:
            jacobian = jac_position[:, self.dofs]
            error = position_error
        else:
            # Scale angular rows into comparable Cartesian units (0.15 m/rad).
            jacobian = np.vstack((jac_position[:, self.dofs], 0.15 * jac_rotation[:, self.dofs]))
            error = np.concatenate((position_error, 0.15 * rotation_error))
        dq = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 0.025**2 * np.eye(jacobian.shape[0]), error
        )
        desired = self.joint_positions + np.clip(dq, -max_joint_step, max_joint_step)
        with self._viewer_lock():
            self.data.ctrl[self.actuators] = np.clip(desired, self.ranges[:, 0], self.ranges[:, 1])

    def _rotation_error(self, target_rotation: np.ndarray | None) -> np.ndarray | None:
        if target_rotation is None:
            return None
        current_rotation = self.data.site_xmat[self.site].reshape(3, 3)
        relative_rotation = np.asarray(target_rotation) @ current_rotation.T
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, relative_rotation.reshape(-1))
        angular_error = np.empty(3)
        mujoco.mju_quat2Vel(angular_error, quaternion, 1.0)
        return angular_error

    def _hold_current_position(self) -> None:
        """Stop residual servo target increments once a waypoint is reached."""
        with self._viewer_lock():
            self.data.ctrl[self.actuators] = np.clip(
                self.joint_positions, self.ranges[:, 0], self.ranges[:, 1]
            )

    def _viewer_lock(self):
        return self.env.viewer.lock() if self.env.viewer is not None else nullcontext()
