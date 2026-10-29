"""Headless MuJoCo integration tests."""

from pathlib import Path

import pytest

from recovera.config import RecoveraConfig
from recovera.environment import MujocoEnvironment
from recovera.robot import PandaAdapter
from recovera.runner import run_trial

ASSET = Path(__file__).parents[1] / "assets/franka_emika_panda/panda.xml"


@pytest.mark.integration
def test_model_reset_and_robot_interface_repeat() -> None:
    env = MujocoEnvironment(RecoveraConfig(model_path=ASSET))
    try:
        robot = PandaAdapter(env)
        q, cube = robot.joint_positions.copy(), robot.cube_position.copy()
        env.step(20)
        env.reset()
        assert robot.joint_positions == pytest.approx(q)
        assert robot.cube_position == pytest.approx(cube)
        assert len(robot.end_effector_position) == 3
    finally:
        env.close()


@pytest.mark.integration
def test_deterministic_end_to_end_pick_and_place() -> None:
    result = run_trial(RecoveraConfig(model_path=ASSET))
    assert result.success, result.model_dump_json(indent=2)
    assert not result.events


@pytest.mark.integration
def test_monitor_extracts_actual_named_model_state_and_scene_change() -> None:
    from recovera.monitoring import TaskExpectation, TaskPhase, WorldStateMonitor

    env = MujocoEnvironment(RecoveraConfig(model_path=ASSET))
    try:
        monitor = WorldStateMonitor(env)
        snapshot = monitor.extractor.capture()
        assert snapshot.valid
        assert snapshot.cube_position == pytest.approx(tuple(env.body_position("cube")))
        assert snapshot.bilateral_finger_contact is False
        env.record_scene_change("test_obstacle", "move")
        events = monitor.observe(TaskExpectation(phase=TaskPhase.APPROACHING))
        assert len(events) == 1 and events[0].event_type.value == "scene_changed"
        assert not monitor.observe(TaskExpectation(phase=TaskPhase.APPROACHING))
    finally:
        env.close()


@pytest.mark.integration
def test_scenario_scene_and_trial_state_reset_cleanly() -> None:
    import mujoco

    env = MujocoEnvironment(RecoveraConfig(model_path=ASSET, scenario_id="obstacle-replan"))
    try:
        obstacle = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "route_obstacle")
        assert obstacle >= 0
        assert int(env.model.geom_contype[obstacle]) == 1
        env.record_scene_change("route_obstacle", "add")
        cube_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        env.data.xfrc_applied[cube_body, 0] = 1.0
        env.reset()
        assert env.scene_revision == 0
        assert env.last_scene_change is None
        assert env.data.xfrc_applied[cube_body, 0] == 0
        assert env.data.geom_xpos[obstacle] == pytest.approx((0.45, 0.06, 0.409 * 0.65))
        lamp_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "table_lamp")
        assert lamp_body >= 0
        assert sum(
            env.model.geom_type[index] == mujoco.mjtGeom.mjGEOM_MESH
            and env.model.geom_bodyid[index] == lamp_body
            for index in range(env.model.ngeom)
        ) == 15
    finally:
        env.close()


@pytest.mark.integration
def test_collision_checked_joint_plan_validates_executes_and_stales_on_scene_update() -> None:
    import numpy as np

    from recovera.motion import (
        PandaJointPlanner,
        PlanStatus,
        TrajectoryValidator,
        execute_trajectory,
    )

    env = MujocoEnvironment(RecoveraConfig(model_path=ASSET))
    try:
        robot = PandaAdapter(env)
        planner = PandaJointPlanner(env, robot, max_iterations=30)
        start = robot.joint_positions
        goal = start.copy()
        goal[0] += 0.005
        plan = planner.plan(start, goal)
        assert plan.status is PlanStatus.SUCCESS, plan.reason
        assert plan.trajectory is not None
        gate = TrajectoryValidator(planner)
        assert gate.validate(plan, start).valid
        completed, reason = execute_trajectory(robot, plan.trajectory)
        assert completed, reason
        assert np.max(np.abs(robot.joint_positions - goal)) < 0.02
        env.record_scene_change("test_obstacle", "move")
        assert not gate.validate(plan, robot.joint_positions).valid
    finally:
        env.close()


@pytest.mark.integration
def test_viewer_closed_during_step_stops_without_advancing_or_hanging() -> None:
    env = MujocoEnvironment(RecoveraConfig(model_path=ASSET))

    class ClosedViewer:
        def is_running(self):
            return False

        def sync(self):
            raise AssertionError("closed viewer must not be synchronized")

    try:
        before = float(env.data.time)
        env.viewer = ClosedViewer()
        env.step(100)
        assert env.viewer_closed
        assert env.data.time == before
    finally:
        env.viewer = None
        env.close()


@pytest.mark.integration
def test_open_viewer_syncs_after_each_existing_simulation_step() -> None:
    from contextlib import nullcontext

    env = MujocoEnvironment(RecoveraConfig(model_path=ASSET))

    class LiveViewer:
        def __init__(self):
            self.times = []

        def is_running(self):
            return True

        def lock(self):
            return nullcontext()

        def sync(self):
            self.times.append(float(env.data.time))

    viewer = LiveViewer()
    try:
        env.viewer = viewer
        env.step(3)
        assert len(viewer.times) == 3
        assert viewer.times == pytest.approx([0.002, 0.004, 0.006])
    finally:
        env.viewer = None
        env.close()


@pytest.mark.integration
def test_obstacle_replan_uses_collision_checked_alternate_ik_and_places_cube() -> None:
    from recovera.llm import MockStrategyProvider
    from recovera.task import PickAndPlaceTask

    config = RecoveraConfig(
        model_path=ASSET,
        scenario_id="obstacle-replan",
        llm_enabled=True,
        llm_provider="mock",
    )
    env = MujocoEnvironment(config)
    try:
        task = PickAndPlaceTask(env, PandaAdapter(env), config)
        task.recovery.llm_provider = MockStrategyProvider()
        result = task.run()
        assert result.success, result.model_dump_json(indent=2)
        assert result.initial_configuration["cube_position"] == [0.45, -0.2, 0.018]
        assert result.target_position == pytest.approx((0.45, 0.32, 0.0))
        attempt = result.recovery_attempts[0]
        assert attempt.llm_proposal_accepted
        assert attempt.strategy.value == "refresh_and_replan"
        assert "RRT path" in attempt.planning_reason
        assert attempt.validation_passed and attempt.execution_completed
        assert result.outcome_classification == "task_success"
    finally:
        env.close()


@pytest.mark.integration
def test_blocked_route_is_collision_verified_and_safe_stopped() -> None:
    from recovera.llm import MockStrategyProvider
    from recovera.task import PickAndPlaceTask

    config = RecoveraConfig(model_path=ASSET, scenario_id="blocked-route", llm_enabled=True, llm_provider="mock")
    env = MujocoEnvironment(config)
    try:
        task = PickAndPlaceTask(env, PandaAdapter(env), config)
        task.recovery.llm_provider = MockStrategyProvider()
        result = task.run()
        assert result.status.value == "safe_stop"
        assert not result.success
        assert result.safe_stop_achieved
        attempt = result.recovery_attempts[0]
        assert attempt.strategy.value == "safe_stop"
        assert "goal configuration is in collision" in attempt.candidate_feasibility["refresh_and_replan"]
    finally:
        env.close()


@pytest.mark.integration
def test_beyond_reach_drop_is_reachability_checked_and_safe_stopped() -> None:
    from recovera.llm import MockStrategyProvider
    from recovera.task import PickAndPlaceTask

    config = RecoveraConfig(model_path=ASSET, scenario_id="beyond-reach", llm_enabled=True, llm_provider="mock")
    env = MujocoEnvironment(config)
    try:
        task = PickAndPlaceTask(env, PandaAdapter(env), config)
        task.recovery.llm_provider = MockStrategyProvider()
        result = task.run()
        assert not result.success and result.safe_stop_achieved
        assert any(event.event_type.value == "object_dropped" for event in result.events)
        attempt = result.recovery_attempts[0]
        assert attempt.strategy.value == "safe_stop"
        assert "IK failed" in attempt.candidate_feasibility["reapproach_and_regrasp"]
        assert attempt.llm_proposal_accepted and attempt.inference_mode == "mock"
    finally:
        env.close()
