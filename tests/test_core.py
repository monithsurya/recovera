"""Unit-level configuration, result and verification tests."""

import pytest

from recovera.config import RecoveraConfig
from recovera.results import TrialResult, TrialStatus
from recovera.verification import is_cube_in_target


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        RecoveraConfig(timestep=0)


def test_result_structure() -> None:
    result = TrialResult(
        status=TrialStatus.FAILURE,
        success=False,
        elapsed_seconds=0.1,
        simulated_seconds=0.2,
        cube_final_position=(0, 0, 0),
        target_position=(1, 1, 0),
    )
    assert result.status is TrialStatus.FAILURE and not result.success


def test_negative_verification() -> None:
    class Env:
        def body_position(self, name: str):
            assert name == "cube"
            return [0.0, 0.0, 0.025]

    assert not is_cube_in_target(Env(), RecoveraConfig())  # type: ignore[arg-type]


def test_missing_model_message(tmp_path) -> None:
    from recovera.environment import MujocoEnvironment

    with pytest.raises(FileNotFoundError, match="Panda model not found"):
        MujocoEnvironment(RecoveraConfig(model_path=tmp_path / "missing.xml"))


def test_monitor_event_models_serialize_to_json() -> None:
    from recovera.monitoring import EventType, MonitoringEvent, TaskPhase

    event = MonitoringEvent(
        event_type=EventType.SCENE_CHANGED,
        step=10,
        sim_time=0.02,
        entity_id="scene",
        expected_phase=TaskPhase.APPROACHING,
        explanation="controlled update",
        evidence={"scene_revision": 1},
    )
    assert event.model_dump(mode="json")["event_type"] == "scene_changed"


def test_failure_detectors_confirm_sustain_and_reset() -> None:
    from types import SimpleNamespace

    from recovera.monitoring import (
        EventType,
        TaskExpectation,
        TaskPhase,
        WorldSnapshot,
        WorldStateMonitor,
    )

    class Extractor:
        def __init__(self):
            self.current = None

        def capture(self):
            return self.current

    env = SimpleNamespace(scene_revision=0)
    monitor = WorldStateMonitor.__new__(WorldStateMonitor)
    monitor.env = env
    monitor.extractor = Extractor()
    monitor.grasp_timeout_steps = 3
    monitor.drop_confirmation_steps = 2
    monitor.reset()

    def sample(step, contact=False, rev=0, valid=True, cube=(0, 0, 0), grip=(0, 0, 0)):
        monitor.extractor.current = WorldSnapshot(
            step=step,
            sim_time=step * 0.002,
            cube_position=cube,
            cube_orientation=(1, 0, 0, 0),
            cube_linear_velocity=(0, 0, 0),
            cube_angular_velocity=(0, 0, 0),
            gripper_position=grip,
            bilateral_finger_contact=contact,
            scene_revision=rev,
            valid=valid,
        )

    grasp = TaskExpectation(phase=TaskPhase.GRASPING, grasp_attempted=True)
    for step in (1, 2, 3):
        sample(step)
        assert not monitor.observe(grasp)
    sample(4)
    assert monitor.observe(grasp)[0].event_type is EventType.FAILED_GRASP

    monitor.reset()
    carry = TaskExpectation(phase=TaskPhase.CARRYING, previously_held=False)
    sample(5, valid=False, cube=(1, 0, 0))
    assert not monitor.observe(carry)
    sample(6, cube=(1, 0, 0))
    assert not monitor.observe(carry)

    monitor.reset()
    sample(10, contact=True)
    monitor.observe(grasp)
    carry = TaskExpectation(phase=TaskPhase.CARRYING, previously_held=True)
    sample(11, cube=(0.2, 0, 0), grip=(0, 0, 0))
    assert not monitor.observe(carry)
    sample(12, contact=True)
    assert not monitor.observe(carry)
    sample(13, cube=(0.2, 0, 0), grip=(0, 0, 0))
    assert not monitor.observe(carry)
    sample(14, cube=(0.2, 0, 0), grip=(0, 0, 0))
    assert not monitor.observe(carry)
    sample(15, cube=(0.2, 0, 0), grip=(0, 0, 0))
    assert monitor.observe(carry)[0].event_type is EventType.OBJECT_DROPPED
    sample(16, cube=(0.2, 0, 0), grip=(0, 0, 0))
    assert not monitor.observe(carry)

    sample(17, rev=1)
    assert monitor.observe(carry)[0].event_type is EventType.SCENE_CHANGED
    sample(18, rev=1)
    assert not monitor.observe(carry)
    env.scene_revision = 1
    monitor.reset()
    sample(19, rev=1)
    assert not monitor.observe(carry)


def test_recovery_policy_requires_confirmed_relevant_event() -> None:
    from types import SimpleNamespace

    from recovera.monitoring import (
        EventType,
        MonitoringEvent,
        TaskExpectation,
        TaskPhase,
        WorldSnapshot,
    )
    from recovera.recovery import RecoveryPolicy, RecoveryStrategy

    event = MonitoringEvent(
        event_type=EventType.OBJECT_DROPPED,
        step=20,
        sim_time=0.04,
        entity_id="cube",
        expected_phase=TaskPhase.CARRYING,
        explanation="loss confirmed",
        evidence={},
    )
    snapshot = WorldSnapshot(
        step=21,
        sim_time=0.042,
        cube_position=(0.4, 0, 0.02),
        cube_orientation=(1, 0, 0, 0),
        cube_linear_velocity=(0, 0, 0),
        cube_angular_velocity=(0, 0, 0),
        gripper_position=(0, 0, 0),
        bilateral_finger_contact=False,
        scene_revision=0,
    )
    task = SimpleNamespace(phase=TaskPhase.FAILED, current_waypoint=None)
    policy = RecoveryPolicy()
    assert (
        policy.choose(
            event,
            snapshot,
            task,
            TaskExpectation(
                phase=TaskPhase.FAILED,
                previously_held=False,
            ),
        )
        is None
    )
    assert (
        policy.choose(
            event,
            snapshot,
            task,
            TaskExpectation(
                phase=TaskPhase.FAILED,
                previously_held=True,
            ),
        )
        is RecoveryStrategy.CENTERED_REAPPROACH_AND_REGRASP
    )


@pytest.mark.integration
def test_recovery_coordinator_rejects_unsupported_scene_route_and_deduplicates() -> None:
    from pathlib import Path
    from types import SimpleNamespace

    from recovera.environment import MujocoEnvironment
    from recovera.monitoring import EventType, MonitoringEvent, TaskPhase, WorldStateExtractor
    from recovera.recovery import RecoveryCoordinator
    from recovera.robot import PandaAdapter

    env = MujocoEnvironment(
        RecoveraConfig(model_path=Path(__file__).parents[1] / "assets/franka_emika_panda/panda.xml")
    )
    try:
        robot = PandaAdapter(env)
        task = SimpleNamespace(
            monitor=SimpleNamespace(extractor=WorldStateExtractor(env)),
            phase=TaskPhase.CARRYING,
            grasp_attempted=True,
            grasp_confirmed=True,
            previously_held=True,
            current_waypoint=None,
        )
        coordinator = RecoveryCoordinator(env, robot, max_attempts=1)
        event = MonitoringEvent(
            event_type=EventType.SCENE_CHANGED,
            step=10,
            sim_time=0.02,
            entity_id="obstacle",
            expected_phase=TaskPhase.CARRYING,
            explanation="controlled obstacle update",
            evidence={"scene_revision": 1},
        )
        first = coordinator.recover(event, task)
        second = coordinator.recover(event, task)
        assert first.terminal_reason == "invalid_state_or_unsupported_strategy"
        assert second.terminal_reason == "duplicate_event"
        assert first.planning_status is None
    finally:
        env.close()


def test_task_phase_logging_deduplicates_transition(caplog) -> None:
    import logging
    from pathlib import Path

    from recovera.environment import MujocoEnvironment
    from recovera.monitoring import TaskPhase
    from recovera.robot import PandaAdapter
    from recovera.task import PickAndPlaceTask

    env = MujocoEnvironment(
        RecoveraConfig(model_path=Path(__file__).parents[1] / "assets/franka_emika_panda/panda.xml")
    )
    try:
        task = PickAndPlaceTask(
            env,
            PandaAdapter(env),
            RecoveraConfig(
                model_path=Path(__file__).parents[1] / "assets/franka_emika_panda/panda.xml"
            ),
        )
        with caplog.at_level(logging.INFO, logger="recovera.SIMULATION"):
            task._set_phase(TaskPhase.APPROACHING)
            task._set_phase(TaskPhase.APPROACHING)
        messages = [r.message for r in caplog.records if "task phase=APPROACHING" in r.message]
        assert len(messages) == 1
    finally:
        env.close()
