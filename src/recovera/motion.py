"""MuJoCo-backed joint-space planning, trajectory validation, and execution."""

from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise

import mujoco
import numpy as np


class PlanStatus(str, Enum):
    SUCCESS = "success"
    INVALID_INPUT = "invalid_input"
    NO_PATH = "no_path"
    ERROR = "error"


@dataclass(frozen=True)
class JointTrajectory:
    points: tuple[tuple[float, ...], ...]
    scene_revision: int
    start: tuple[float, ...]
    goal: tuple[float, ...]
    allow_final_finger_contact: bool = False


@dataclass(frozen=True)
class PlanResult:
    status: PlanStatus
    trajectory: JointTrajectory | None
    reason: str


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str


class PandaJointPlanner:
    """Deterministic RRT planner with MuJoCo collision checks on Panda collision geoms."""

    def __init__(self, env, robot, *, seed: int = 17, max_iterations: int = 1600):
        self.env, self.robot = env, robot
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.max_iterations = max_iterations
        self.model = env.model
        self.data = mujoco.MjData(env.model)
        self.qpos_addresses = np.asarray(robot.qpos, dtype=int)
        self.joint_ids = np.asarray(robot.joints, dtype=int)
        self.lower = self.model.jnt_range[self.joint_ids, 0].copy()
        self.upper = self.model.jnt_range[self.joint_ids, 1].copy()
        self.arm_geoms = {
            geom for geom in range(self.model.ngeom) if int(self.model.geom_contype[geom]) & 1
        }
        self.cube_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.cube_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.site_id = robot.site
        self.carrying_payload = robot.has_grasp_contact()
        self.payload_offset_local = None
        self.payload_rotation_local = None
        if self.carrying_payload:
            site_rotation = env.data.site_xmat[self.site_id].reshape(3, 3).copy()
            cube_rotation = env.data.xmat[self.cube_body].reshape(3, 3).copy()
            self.payload_offset_local = site_rotation.T @ (
                env.data.xpos[self.cube_body] - env.data.site_xpos[self.site_id]
            )
            self.payload_rotation_local = site_rotation.T @ cube_rotation
        finger_bodies = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("left_finger", "right_finger")
        }
        self.finger_geoms = {
            geom for geom in self.arm_geoms if int(self.model.geom_bodyid[geom]) in finger_bodies
        }

    def reset(self) -> None:
        """Restore the configured planner seed between reproducible trials."""
        self.rng = np.random.default_rng(self.seed)

    def _valid_configuration(
        self, q: np.ndarray, *, allow_initial_finger_contact: bool = False
    ) -> bool:
        if q.shape != (len(self.joint_ids),) or not np.isfinite(q).all():
            return False
        if np.any(q < self.lower) or np.any(q > self.upper):
            return False
        self.data.qpos[:] = self.env.data.qpos
        self.data.qvel[:] = 0
        self.data.qpos[self.qpos_addresses] = q
        mujoco.mj_forward(self.model, self.data)
        if self.carrying_payload:
            site_rotation = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
            cube_position = self.data.site_xpos[self.site_id] + site_rotation @ self.payload_offset_local
            cube_rotation = site_rotation @ self.payload_rotation_local
            cube_qpos = int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")])
            self.data.qpos[cube_qpos : cube_qpos + 3] = cube_position
            cube_quaternion = np.empty(4)
            mujoco.mju_mat2Quat(cube_quaternion, cube_rotation.reshape(-1))
            self.data.qpos[cube_qpos + 3 : cube_qpos + 7] = cube_quaternion
            mujoco.mj_forward(self.model, self.data)
        # Panda collision geoms use contype bit 1. Any resulting contact involving
        # one of these geoms is rejected; cube-floor support is therefore ignored.
        for i in range(self.data.ncon):
            first, second = int(self.data.contact[i].geom1), int(self.data.contact[i].geom2)
            if first not in self.arm_geoms and second not in self.arm_geoms:
                continue
            permitted_start_contact = (allow_initial_finger_contact or self.carrying_payload) and (
                (first == self.cube_geom and second in self.finger_geoms)
                or (second == self.cube_geom and first in self.finger_geoms)
            )
            if not permitted_start_contact:
                return False
        return True

    def _edge_valid(
        self,
        a: np.ndarray,
        b: np.ndarray,
        resolution: float = 0.045,
        *,
        allow_final_finger_contact: bool = False,
    ) -> bool:
        count = max(1, int(np.ceil(np.max(np.abs(b - a)) / resolution)))
        return all(
            self._valid_configuration(
                a + (b - a) * (i / count),
                allow_initial_finger_contact=allow_final_finger_contact and i == count,
            )
            for i in range(1, count + 1)
        )

    def plan(
        self,
        start_state,
        goal,
        *,
        scene_revision: int | None = None,
        allow_initial_finger_contact: bool = True,
        allow_goal_finger_contact: bool = False,
    ) -> PlanResult:
        start, target = np.asarray(start_state, dtype=float), np.asarray(goal, dtype=float)
        if (
            start.shape != (7,)
            or target.shape != (7,)
            or not np.isfinite(start).all()
            or not np.isfinite(target).all()
        ):
            return PlanResult(
                PlanStatus.INVALID_INPUT, None, "start and goal must be finite 7-joint vectors"
            )
        if np.any(target < self.lower) or np.any(target > self.upper):
            return PlanResult(PlanStatus.INVALID_INPUT, None, "goal violates Panda joint limits")
        try:
            if not self._valid_configuration(
                start, allow_initial_finger_contact=allow_initial_finger_contact
            ):
                return PlanResult(
                    PlanStatus.INVALID_INPUT, None, "current robot configuration is in collision"
                )
            if not self._valid_configuration(
                target, allow_initial_finger_contact=allow_goal_finger_contact
            ):
                return PlanResult(PlanStatus.NO_PATH, None, "goal configuration is in collision")
            direct_valid = self._edge_valid(
                start, target, allow_final_finger_contact=allow_goal_finger_contact
            )
            used_rrt = (not direct_valid) or self.env.config.scenario_id == "obstacle-replan"
            if not used_rrt:
                path = [start, target]
            else:
                path = self._rrt(start, target, allow_goal_finger_contact=allow_goal_finger_contact)
                if path is None:
                    return PlanResult(
                        PlanStatus.NO_PATH, None, "RRT did not find a collision-free path"
                    )
            points = self._time_parameterize(path)
            traj = JointTrajectory(
                points=tuple(tuple(float(v) for v in row) for row in points),
                scene_revision=self.env.scene_revision
                if scene_revision is None
                else scene_revision,
                start=tuple(float(v) for v in start),
                goal=tuple(float(v) for v in target),
                allow_final_finger_contact=allow_goal_finger_contact,
            )
            reason = (
                "collision-checked MuJoCo RRT path (straight edge rejected)"
                if used_rrt and not direct_valid
                else "collision-checked MuJoCo RRT path"
                if used_rrt
                else "collision-checked MuJoCo direct path"
            )
            return PlanResult(PlanStatus.SUCCESS, traj, reason)
        except (ValueError, mujoco.FatalError, np.linalg.LinAlgError) as exc:
            return PlanResult(PlanStatus.ERROR, None, f"planning error: {exc}")

    def _rrt(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        *,
        allow_goal_finger_contact: bool = False,
    ) -> list[np.ndarray] | None:
        # Try a bounded deterministic joint-space detour before randomized
        # exploration. This is effective for the small, known obstacles in the
        # Recovera tabletop scenes and keeps routes reproducible.
        for joint in range(7):
            for magnitude in (0.08, 0.16, 0.24, 0.32, 0.48):
                for sign in (-1.0, 1.0):
                    first = start + 0.35 * (goal - start)
                    second = start + 0.68 * (goal - start)
                    first[joint] += sign * magnitude
                    second[joint] += sign * magnitude
                    if (
                        np.all(first >= self.lower)
                        and np.all(first <= self.upper)
                        and np.all(second >= self.lower)
                        and np.all(second <= self.upper)
                        and self._edge_valid(start, first)
                        and self._edge_valid(first, second)
                        and self._edge_valid(
                            second,
                            goal,
                            allow_final_finger_contact=allow_goal_finger_contact,
                        )
                    ):
                        return [start.copy(), first, second, goal.copy()]
        trees = [
            {"nodes": [start.copy()], "parents": [-1]},
            {"nodes": [goal.copy()], "parents": [-1]},
        ]
        active_is_start = True

        def add_node(tree, point, parent):
            tree["nodes"].append(point.copy())
            tree["parents"].append(parent)
            return len(tree["nodes"]) - 1

        def chain(tree, index):
            result = []
            while index >= 0:
                result.append(tree["nodes"][index])
                index = tree["parents"][index]
            return list(reversed(result))

        def extend(tree, target):
            nodes = tree["nodes"]
            nearest = int(np.argmin([np.linalg.norm(node - target) for node in nodes]))
            current = nodes[nearest]
            distance = float(np.linalg.norm(target - current))
            if distance < 1e-9:
                return nearest, True
            candidate = current + (target - current) * min(0.24, distance) / distance
            if not self._edge_valid(current, candidate):
                return nearest, False
            new_index = add_node(tree, candidate, nearest)
            if distance <= 0.24 and self._edge_valid(
                candidate,
                target,
                allow_final_finger_contact=allow_goal_finger_contact and active_is_start,
            ):
                target_index = add_node(tree, target, new_index)
                return target_index, True
            return new_index, False

        for iteration in range(self.max_iterations):
            sample = goal if iteration % 8 == 0 else self.rng.uniform(self.lower, self.upper)
            active_index = 0 if active_is_start else 1
            other_index = 1 - active_index
            active, other = trees[active_index], trees[other_index]
            new_index, _ = extend(active, sample)
            target = active["nodes"][new_index]
            connection_index, connected = extend(other, target)
            if connected and np.linalg.norm(other["nodes"][connection_index] - target) < 1e-9:
                active_path = chain(active, new_index)
                other_path = chain(other, connection_index)
                if active_is_start:
                    return active_path + list(reversed(other_path))[1:]
                return other_path + list(reversed(active_path))[1:]
            active_is_start = not active_is_start
        return None

    @staticmethod
    def _time_parameterize(
        path: list[np.ndarray], max_joint_step: float = 0.025
    ) -> list[np.ndarray]:
        trajectory = [path[0].copy()]
        for a, b in pairwise(path):
            segments = max(1, int(np.ceil(np.max(np.abs(b - a)) / max_joint_step)))
            trajectory.extend(a + (b - a) * (i / segments) for i in range(1, segments + 1))
        return trajectory


class TrajectoryValidator:
    """Recheck trajectory shape, start consistency, limits, freshness and collision."""

    def __init__(self, planner: PandaJointPlanner, start_tolerance: float = 0.06):
        self.planner = planner
        self.start_tolerance = start_tolerance

    def validate(self, plan: PlanResult, current_state) -> ValidationResult:
        trajectory = plan.trajectory
        if plan.status is not PlanStatus.SUCCESS or trajectory is None:
            return ValidationResult(False, f"planner did not succeed: {plan.reason}")
        if trajectory.scene_revision != self.planner.env.scene_revision:
            return ValidationResult(False, "scene changed after planning; replan required")
        points = np.asarray(trajectory.points, dtype=float)
        current = np.asarray(current_state, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (7,) or len(points) < 2:
            return ValidationResult(
                False, "trajectory must contain at least two seven-joint points"
            )
        if not np.isfinite(points).all() or current.shape != (7,) or not np.isfinite(current).all():
            return ValidationResult(False, "trajectory or current state contains invalid values")
        if np.max(np.abs(points[0] - current)) > self.start_tolerance:
            return ValidationResult(False, "trajectory start does not match current robot state")
        for index, point in enumerate(points):
            if not self.planner._valid_configuration(
                point,
                allow_initial_finger_contact=(index == 0)
                or (trajectory.allow_final_finger_contact and index == len(points) - 1),
            ):
                return ValidationResult(
                    False, "trajectory violates joint limits or collision constraints"
                )
        edges = tuple(pairwise(points))
        if any(
            not self.planner._edge_valid(
                a,
                b,
                allow_final_finger_contact=(
                    trajectory.allow_final_finger_contact and index == len(edges) - 1
                ),
            )
            for index, (a, b) in enumerate(edges)
        ):
            return ValidationResult(False, "trajectory segment failed collision recheck")
        return ValidationResult(True, "trajectory passed current-scene checks")


def solve_site_ik(
    robot, target_position, target_rotation=None, *, initial_joint_state=None
) -> np.ndarray | None:
    """Solve a Panda site pose on private MuJoCo data; leave the live controller untouched."""
    model = robot.model
    data = mujoco.MjData(model)
    data.qpos[:] = robot.data.qpos
    qpos = np.asarray(robot.qpos, dtype=int)
    joint_ids = np.asarray(robot.joints, dtype=int)
    lower = model.jnt_range[joint_ids, 0]
    upper = model.jnt_range[joint_ids, 1]
    if initial_joint_state is not None:
        initial = np.asarray(initial_joint_state, dtype=float)
        if initial.shape != (7,) or not np.isfinite(initial).all():
            return None
        data.qpos[qpos] = initial
    data.qvel[:] = 0
    dofs = np.asarray(robot.dofs, dtype=int)
    goal = np.asarray(target_position, dtype=float)
    if goal.shape != (3,) or not np.isfinite(goal).all():
        return None
    rotation_goal = None if target_rotation is None else np.asarray(target_rotation, dtype=float)
    for _ in range(300):
        mujoco.mj_forward(model, data)
        position_error = goal - data.site_xpos[robot.site]
        jac_position = np.zeros((3, model.nv))
        jac_rotation = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac_position, jac_rotation, robot.site)
        if rotation_goal is None:
            error, jacobian = position_error, jac_position[:, dofs]
            rotation_norm = 0.0
        else:
            relative = rotation_goal @ data.site_xmat[robot.site].reshape(3, 3).T
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, relative.reshape(-1))
            rotation_error = np.empty(3)
            mujoco.mju_quat2Vel(rotation_error, quaternion, 1.0)
            rotation_norm = float(np.linalg.norm(rotation_error))
            error = np.concatenate((position_error, 0.15 * rotation_error))
            jacobian = np.vstack((jac_position[:, dofs], 0.15 * jac_rotation[:, dofs]))
        if np.linalg.norm(position_error) < 0.004 and rotation_norm < 0.06:
            return data.qpos[qpos].copy()
        dq = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 0.025**2 * np.eye(jacobian.shape[0]), error
        )
        data.qpos[qpos] = np.clip(data.qpos[qpos] + np.clip(dq, -0.035, 0.035), lower, upper)
    return None


def execute_trajectory(
    robot, trajectory: JointTrajectory, *, tolerance: float = 0.012
) -> tuple[bool, str]:
    """Execute through Panda's existing position-servo actuators, synchronously."""
    if trajectory.scene_revision != robot.env.scene_revision:
        return False, "scene changed before trajectory execution"
    if np.max(np.abs(robot.joint_positions - np.asarray(trajectory.start))) > 0.06:
        return False, "robot state no longer matches the validated trajectory start"
    data = robot.data
    for point in trajectory.points[1:]:
        target = np.asarray(point)
        lock = robot.env.viewer.lock() if robot.env.viewer is not None else nullcontext()
        with lock:
            data.ctrl[robot.actuators] = np.clip(target, robot.ranges[:, 0], robot.ranges[:, 1])
        for _ in range(150):
            robot.env.step()
            if robot.env.viewer_closed:
                return False, "viewer closed during trajectory execution"
            if np.max(np.abs(robot.joint_positions - target)) <= tolerance:
                break
        else:
            return False, "joint servo failed to track a planned waypoint"
        if trajectory.scene_revision != robot.env.scene_revision:
            return False, "scene changed during trajectory execution"
    robot._hold_current_position()
    return True, "trajectory executed through Panda position servos"
