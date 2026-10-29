"""Offline LLM contract, fallback, and MuJoCo integration tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from recovera.config import RecoveraConfig
from recovera.llm import (
    LLMProviderTimeout,
    MockStrategyProvider,
    RecoveryRequest,
    StrategyCandidate,
    StrategyProposal,
    validate_proposal,
)
from recovera.monitoring import (
    EventType,
    MonitoringEvent,
    TaskExpectation,
    TaskPhase,
    WorldSnapshot,
)
from recovera.recovery import RecoveryCoordinator, RecoveryPolicy, RecoveryStrategy

ASSET = Path(__file__).parents[1] / "assets/franka_emika_panda/panda.xml"


def make_request() -> RecoveryRequest:
    return RecoveryRequest(
        request_id="req-1",
        task_phase=TaskPhase.GRASPING,
        original_task_goal_xyz=(0.45, 0.25, 0.0),
        failure={
            "event_type": EventType.FAILED_GRASP,
            "step": 12,
            "sim_time": 0.024,
            "entity_id": "cube",
            "explanation": "no bilateral grasp",
        },
        observed_step=13,
        observed_sim_time=0.026,
        cube_position_xyz=(0.45, 0, 0.018),
        cube_orientation_wxyz=(1, 0, 0, 0),
        bilateral_finger_contact=False,
        robot_joint_positions_rad=(0, 0, 0, -1.5, 0, 1.5, -0.7),
        scene_revision=0,
        state_valid=True,
        prerequisites={
            "world_state_valid": True,
            "grasp_attempted": True,
            "object_previously_held": False,
            "current_waypoint_available": False,
        },
        candidates=(
            StrategyCandidate(
                strategy_id="reapproach_and_regrasp",
                target_subgoal="reacquire_cube_calibrated_offset",
                description="Use the calibrated offset.",
                prerequisites=("grasp_attempted",),
                requires_replanning=True,
            ),
            StrategyCandidate(
                strategy_id="centered_reapproach_and_regrasp",
                target_subgoal="reacquire_cube_centered",
                description="Use a centered approach.",
                prerequisites=("grasp_attempted",),
                requires_replanning=True,
            ),
        ),
        previous_attempts=(),
    )


def valid_proposal(
    request: RecoveryRequest, strategy_id: str = "reapproach_and_regrasp"
) -> StrategyProposal:
    candidate = next(item for item in request.candidates if item.strategy_id == strategy_id)
    return StrategyProposal(
        request_id=request.request_id,
        strategy_id=candidate.strategy_id,
        target_subgoal=candidate.target_subgoal,
        prerequisites=candidate.prerequisites,
        rationale="Retry the confirmed grasp subgoal.",
        requires_replanning=candidate.requires_replanning,
    )


def test_strategy_contract_accepts_only_bounded_consistent_output() -> None:
    request = make_request()
    valid = validate_proposal(request, valid_proposal(request))
    assert valid.accepted

    unsupported = valid_proposal(request).model_copy(update={"strategy_id": "move_anywhere"})
    assert not validate_proposal(request, unsupported).accepted
    stale = valid_proposal(request).model_copy(update={"request_id": "old-request"})
    assert "stale" in validate_proposal(request, stale).reason
    wrong_subgoal = valid_proposal(request).model_copy(update={"target_subgoal": "invent_pose"})
    assert not validate_proposal(request, wrong_subgoal).accepted
    contradictory_precondition = valid_proposal(request).model_copy(
        update={"prerequisites": ("current_waypoint_available",)}
    )
    assert not validate_proposal(request, contradictory_precondition).accepted
    skips_planner = valid_proposal(request).model_copy(update={"requires_replanning": False})
    assert not validate_proposal(request, skips_planner).accepted
    with pytest.raises(ValueError):
        StrategyProposal.model_validate(
            {**valid_proposal(request).model_dump(), "joint_angles": [0]}
        )


class FakeProvider:
    provider_name = "mock"
    model_name = "mock-model"
    timeout_seconds = 1.0

    def __init__(self, callback):
        self.callback = callback
        self.requests = []

    def propose(self, request, *, timeout_seconds=None):
        self.requests.append(request)
        return self.callback(request)


def make_selector(provider, *, fallback="deterministic", capture=None):
    coordinator = RecoveryCoordinator.__new__(RecoveryCoordinator)
    coordinator.env = SimpleNamespace(last_scene_change=None)
    coordinator.robot = SimpleNamespace(joint_positions=(0, 0, 0, -1.5, 0, 1.5, -0.7))
    coordinator.llm_provider = provider
    coordinator.llm_enabled = True
    coordinator.llm_fallback = fallback
    coordinator.llm_total_budget_seconds = 2.0
    coordinator._llm_elapsed = 0.0
    coordinator._accepted_proposals = set()
    coordinator.policy = RecoveryPolicy()
    request_snapshot = WorldSnapshot(
        step=13,
        sim_time=0.026,
        cube_position=(0.45, 0, 0.018),
        cube_orientation=(1, 0, 0, 0),
        cube_linear_velocity=(0, 0, 0),
        cube_angular_velocity=(0, 0, 0),
        gripper_position=(0.45, 0, 0.2),
        bilateral_finger_contact=False,
        scene_revision=0,
    )
    expectation = TaskExpectation(
        phase=TaskPhase.GRASPING,
        grasp_attempted=True,
        previously_held=False,
    )
    task = SimpleNamespace(
        phase=TaskPhase.GRASPING,
        grasp_attempted=True,
        grasp_confirmed=False,
        previously_held=False,
        current_waypoint=None,
        config=SimpleNamespace(target_position=(0.45, 0.25, 0.0)),
        recovery_attempts=[],
        monitor=SimpleNamespace(
            extractor=SimpleNamespace(capture=capture or (lambda: request_snapshot))
        ),
    )
    event = MonitoringEvent(
        event_type=EventType.FAILED_GRASP,
        step=12,
        sim_time=0.024,
        entity_id="cube",
        expected_phase=TaskPhase.GRASPING,
        explanation="no bilateral grasp",
        evidence={},
    )
    return coordinator, event, request_snapshot, task, expectation


@pytest.mark.parametrize("fallback", ["deterministic", "stop"])
def test_timeout_uses_configured_fallback(fallback) -> None:
    provider = FakeProvider(lambda request: (_ for _ in ()).throw(LLMProviderTimeout("timeout")))
    coordinator, event, snapshot, task, expectation = make_selector(provider, fallback=fallback)
    strategy, _, _, trace = coordinator._llm_select(
        event, snapshot, task, expectation, RecoveryStrategy.REAPPROACH_AND_REGRASP
    )
    expected = RecoveryStrategy.REAPPROACH_AND_REGRASP if fallback == "deterministic" else None
    assert strategy is expected
    assert not trace["accepted"]
    assert "timeout" in trace["fallback"]
    assert coordinator._llm_elapsed >= 0


def test_malformed_and_unsupported_proposals_fall_back_without_execution() -> None:
    for response in (
        {"strategy_id": "reapproach_and_regrasp"},
        valid_proposal(make_request()).model_copy(update={"strategy_id": "raw_joint_command"}),
    ):
        provider = FakeProvider(lambda request, output=response: output)
        coordinator, event, snapshot, task, expectation = make_selector(provider)
        strategy, _, _, trace = coordinator._llm_select(
            event, snapshot, task, expectation, RecoveryStrategy.REAPPROACH_AND_REGRASP
        )
        assert strategy is RecoveryStrategy.REAPPROACH_AND_REGRASP
        assert not trace["accepted"]


def test_stale_proposal_is_rejected_and_deterministic_policy_uses_fresh_state() -> None:
    request_snapshot = make_selector(FakeProvider(lambda _: None))[2]
    changed = request_snapshot.model_copy(update={"step": request_snapshot.step + 1})
    provider = FakeProvider(valid_proposal)
    coordinator, event, snapshot, task, expectation = make_selector(
        provider, capture=lambda: changed
    )
    strategy, current, _, trace = coordinator._llm_select(
        event, snapshot, task, expectation, RecoveryStrategy.REAPPROACH_AND_REGRASP
    )
    assert current.step == changed.step
    assert strategy is None
    assert not trace["accepted"]
    assert "stale" in trace["reason"]


def test_missing_provider_and_exhausted_budget_are_bounded() -> None:
    coordinator, event, snapshot, task, expectation = make_selector(None)
    strategy, _, _, trace = coordinator._llm_select(
        event, snapshot, task, expectation, RecoveryStrategy.REAPPROACH_AND_REGRASP
    )
    assert strategy is RecoveryStrategy.REAPPROACH_AND_REGRASP
    assert "no LLM provider" in trace["fallback"]

    provider = FakeProvider(valid_proposal)
    coordinator, event, snapshot, task, expectation = make_selector(provider)
    coordinator._llm_elapsed = coordinator.llm_total_budget_seconds
    strategy, _, _, trace = coordinator._llm_select(
        event, snapshot, task, expectation, RecoveryStrategy.REAPPROACH_AND_REGRASP
    )
    assert strategy is RecoveryStrategy.REAPPROACH_AND_REGRASP
    assert not provider.requests
    assert "budget exhausted" in trace["fallback"]


def test_repeated_proposal_is_rejected_and_uses_bounded_fallback() -> None:
    provider = FakeProvider(valid_proposal)
    coordinator, event, snapshot, task, expectation = make_selector(provider)
    first, _, _, first_trace = coordinator._llm_select(
        event, snapshot, task, expectation, RecoveryStrategy.REAPPROACH_AND_REGRASP
    )
    second, _, _, second_trace = coordinator._llm_select(
        event, snapshot, task, expectation, RecoveryStrategy.REAPPROACH_AND_REGRASP
    )
    assert first is RecoveryStrategy.REAPPROACH_AND_REGRASP and first_trace["accepted"]
    assert second is RecoveryStrategy.REAPPROACH_AND_REGRASP and not second_trace["accepted"]
    assert "repeated proposal" in second_trace["reason"]


def test_provider_can_select_a_second_genuinely_permitted_candidate() -> None:
    provider = FakeProvider(
        lambda request: valid_proposal(request, "centered_reapproach_and_regrasp")
    )
    coordinator, event, snapshot, task, expectation = make_selector(provider)
    candidates = coordinator.policy.candidates(event, snapshot, task, expectation)
    selected, _, _, trace = coordinator._llm_select(
        event,
        snapshot,
        task,
        expectation,
        RecoveryStrategy.REAPPROACH_AND_REGRASP,
        candidates,
    )
    assert tuple(item.strategy_id for item in candidates) == (
        "reapproach_and_regrasp",
        "centered_reapproach_and_regrasp",
    )
    assert selected is RecoveryStrategy.CENTERED_REAPPROACH_AND_REGRASP
    assert trace["accepted"]
    assert selected.value != RecoveryStrategy.REAPPROACH_AND_REGRASP.value


def test_explicit_mock_provider_is_labeled_and_does_not_call_ollama() -> None:
    request = make_request()
    provider = MockStrategyProvider("centered_reapproach_and_regrasp")
    proposal = provider.propose(request)
    assert provider.inference_mode == "mock"
    assert proposal.strategy_id == "centered_reapproach_and_regrasp"
    assert validate_proposal(request, proposal).accepted


def test_explicit_abstention_stops_instead_of_using_deterministic_fallback() -> None:
    def abstain(request):
        return StrategyProposal(
            request_id=request.request_id,
            strategy_id=None,
            target_subgoal=None,
            prerequisites=(),
            rationale="State is too uncertain to choose safely.",
            requires_replanning=False,
            abstain=True,
        )

    provider = FakeProvider(abstain)
    coordinator, event, snapshot, task, expectation = make_selector(provider)
    candidates = coordinator.policy.candidates(event, snapshot, task, expectation)
    strategy, _, _, trace = coordinator._llm_select(
        event,
        snapshot,
        task,
        expectation,
        RecoveryStrategy.REAPPROACH_AND_REGRASP,
        candidates,
    )
    assert strategy is None
    assert trace["abstain"]


@pytest.mark.integration
def test_mock_llm_recovery_runs_through_real_panda_planner_and_verifier() -> None:
    from recovera.environment import MujocoEnvironment
    from recovera.llm import StrategyProposal
    from recovera.robot import PandaAdapter
    from recovera.task import PickAndPlaceTask

    env = MujocoEnvironment(RecoveraConfig(model_path=ASSET, demo_fail_first_grasp=True))
    try:
        robot = PandaAdapter(env)
        task = PickAndPlaceTask(
            env,
            robot,
            RecoveraConfig(
                model_path=ASSET,
                demo_fail_first_grasp=True,
                llm_enabled=True,
            ),
        )

        class TaskMockProvider:
            provider_name = "mock"
            model_name = "fixed-response"
            inference_mode = "mock"
            timeout_seconds = 1.0

            def __init__(self):
                self.called = False

            def propose(self, request, *, timeout_seconds=None):
                self.called = True
                selected = next(
                    candidate
                    for candidate in request.candidates
                    if candidate.strategy_id == "centered_reapproach_and_regrasp"
                )
                return StrategyProposal(
                    request_id=request.request_id,
                    strategy_id=selected.strategy_id,
                    target_subgoal=selected.target_subgoal,
                    prerequisites=selected.prerequisites,
                    rationale="Use the centered permitted regrasp approach.",
                    requires_replanning=selected.requires_replanning,
                )

        mock_provider = TaskMockProvider()
        task.recovery.llm_provider = mock_provider
        result = task.run()
        assert result.success, result.model_dump_json(indent=2)
        assert len(result.recovery_attempts) == 1
        attempt = result.recovery_attempts[0]
        assert attempt.llm_proposal_accepted is True
        assert attempt.strategy is RecoveryStrategy.CENTERED_REAPPROACH_AND_REGRASP
        assert attempt.candidate_strategy_ids == (
            "reapproach_and_regrasp",
            "centered_reapproach_and_regrasp",
        )
        assert attempt.inference_mode == "mock"
        assert attempt.planning_status.value == "success"
        assert attempt.validation_passed and attempt.execution_completed
        assert attempt.verification == "bilateral grasp confirmed"
        assert attempt.resumed_original_task
    finally:
        env.close()


@pytest.mark.integration
def test_failed_first_recovery_uses_one_bounded_permitted_alternative(monkeypatch) -> None:
    from recovera.environment import MujocoEnvironment
    from recovera.llm import MockStrategyProvider
    from recovera.motion import PlanResult, PlanStatus
    from recovera.robot import PandaAdapter
    from recovera.task import PickAndPlaceTask

    config = RecoveraConfig(model_path=ASSET, demo_fail_first_grasp=True, llm_enabled=True)
    env = MujocoEnvironment(config)
    try:
        robot = PandaAdapter(env)
        task = PickAndPlaceTask(env, robot, config)
        task.recovery.llm_provider = MockStrategyProvider()
        planner = task.recovery.planner
        planned = planner.plan
        calls = 0

        def fail_first_plan(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return PlanResult(PlanStatus.NO_PATH, None, "deterministic first-plan rejection")
            return planned(*args, **kwargs)

        monkeypatch.setattr(planner, "plan", fail_first_plan)
        result = task.run()
        assert result.success, result.model_dump_json(indent=2)
        attempts = result.recovery_attempts
        assert len(attempts) == 1
        assert attempts[0].strategy is RecoveryStrategy.CENTERED_REAPPROACH_AND_REGRASP
        assert "pregrasp plan failed" in attempts[0].candidate_feasibility["reapproach_and_regrasp"]
        assert (
            "passed collision/limit validation"
            in attempts[0].candidate_feasibility["centered_reapproach_and_regrasp"]
        )
        assert attempts[0].validation_passed and attempts[0].execution_completed
        assert attempts[0].verification == "bilateral grasp confirmed"
        assert attempts[0].resumed_original_task
        assert calls == 3
    finally:
        env.close()


@pytest.mark.integration
def test_confirmed_drop_recovery_uses_alternative_after_first_regrasp_fails() -> None:
    from recovera.environment import MujocoEnvironment
    from recovera.llm import MockStrategyProvider
    from recovera.robot import PandaAdapter
    from recovera.task import PickAndPlaceTask

    config = RecoveraConfig(model_path=ASSET, demo_drop_after_grasp=True, llm_enabled=True)
    env = MujocoEnvironment(config)
    try:
        task = PickAndPlaceTask(env, PandaAdapter(env), config)
        task.recovery.llm_provider = MockStrategyProvider()
        result = task.run()
        assert result.success, result.model_dump_json(indent=2)
        assert [event.event_type for event in result.events] == [EventType.OBJECT_DROPPED]
        assert len(result.recovery_attempts) == 2
        first, second = result.recovery_attempts
        assert first.strategy is RecoveryStrategy.REAPPROACH_AND_REGRASP
        assert first.terminal_reason == "regrasp_verification_failed"
        assert second.strategy is RecoveryStrategy.CENTERED_REAPPROACH_AND_REGRASP
        assert second.validation_passed and second.execution_completed
        assert second.verification == "bilateral grasp confirmed"
        assert second.resumed_original_task
        assert second.trial_id is None
        assert second.scenario == "drop-demo"
        assert first.outcome_classification == "recovery_failure"
        assert second.outcome_classification == "recovery_success"
        assert result.outcome_classification == "task_success"
    finally:
        env.close()


@pytest.mark.integration
def test_accepted_llm_strategy_stops_when_planner_finds_no_path(monkeypatch) -> None:
    from recovera.environment import MujocoEnvironment
    from recovera.llm import StrategyProposal
    from recovera.motion import PlanResult, PlanStatus
    from recovera.robot import PandaAdapter
    from recovera.task import PickAndPlaceTask

    env = MujocoEnvironment(RecoveraConfig(model_path=ASSET))
    try:
        robot = PandaAdapter(env)
        task = PickAndPlaceTask(env, robot, RecoveraConfig(model_path=ASSET, llm_enabled=True))
        assert robot.initialize()
        task.phase = TaskPhase.GRASPING
        task.grasp_attempted = True

        class TaskMockProvider:
            provider_name = "mock"
            model_name = "fixed-response"
            inference_mode = "mock"
            timeout_seconds = 1.0

            def __init__(self):
                self.called = False

            def propose(self, request, *, timeout_seconds=None):
                self.called = True
                selected = next(
                    candidate
                    for candidate in request.candidates
                    if candidate.strategy_id == "reapproach_and_regrasp"
                )
                return StrategyProposal(
                    request_id=request.request_id,
                    strategy_id=selected.strategy_id,
                    target_subgoal=selected.target_subgoal,
                    prerequisites=selected.prerequisites,
                    rationale="Use the permitted regrasp strategy.",
                    requires_replanning=selected.requires_replanning,
                )

        class RejectingPlanner:
            def plan(self, *args, **kwargs):
                return PlanResult(PlanStatus.NO_PATH, None, "blocked test route")

        task.recovery.planner = RejectingPlanner()
        task.recovery.validator.planner = task.recovery.planner
        mock_provider = TaskMockProvider()
        task.recovery.llm_provider = mock_provider
        monkeypatch.setattr(
            "recovera.recovery.solve_site_ik",
            lambda robot, position, rotation: robot.joint_positions,
        )
        observed = task.monitor.extractor.capture()
        event = MonitoringEvent(
            event_type=EventType.FAILED_GRASP,
            step=observed.step,
            sim_time=observed.sim_time,
            entity_id="cube",
            expected_phase=TaskPhase.GRASPING,
            explanation="confirmed failed grasp",
            evidence={},
        )
        attempt = task.recovery.recover(event, task)
        assert attempt.llm_proposal_accepted is None
        assert attempt.strategy is None
        assert attempt.planning_status is PlanStatus.NO_PATH
        assert attempt.terminal_reason == "no_feasible_candidate"
        assert not mock_provider.called
        assert not attempt.validation_passed
        assert not attempt.execution_completed
        assert not attempt.resumed_original_task
    finally:
        env.close()


@pytest.mark.integration
def test_post_selection_trajectory_revalidation_blocks_execution(monkeypatch) -> None:
    from recovera.environment import MujocoEnvironment
    from recovera.llm import MockStrategyProvider
    from recovera.motion import ValidationResult
    from recovera.robot import PandaAdapter
    from recovera.task import PickAndPlaceTask

    config = RecoveraConfig(
        model_path=ASSET,
        demo_fail_first_grasp=True,
        max_recovery_attempts=0,
        llm_enabled=True,
    )
    env = MujocoEnvironment(config)
    try:
        task = PickAndPlaceTask(env, PandaAdapter(env), config)
        task.run()
        event = next(
            event for event in task.monitor.events if event.event_type is EventType.FAILED_GRASP
        )
        task.recovery_attempts.clear()
        task.recovery.reset()
        task.recovery.max_attempts = 1
        task.recovery.llm_provider = MockStrategyProvider()
        original_validate = task.recovery.validator.validate
        calls = 0

        def reject_selected(plan, current_state):
            nonlocal calls
            calls += 1
            if calls == 5:  # Four candidate preflight checks, then selected-path revalidation.
                return ValidationResult(False, "test state invalidated selected trajectory")
            return original_validate(plan, current_state)

        monkeypatch.setattr(task.recovery.validator, "validate", reject_selected)

        def execution_must_not_run(*args, **kwargs):
            raise AssertionError("rejected trajectory must never reach the execution adapter")

        monkeypatch.setattr("recovera.recovery.execute_trajectory", execution_must_not_run)
        attempt = task.recovery.recover(event, task)
        assert attempt.llm_proposal_accepted is True
        assert attempt.strategy is RecoveryStrategy.REAPPROACH_AND_REGRASP
        assert attempt.terminal_reason == "validation_failed"
        assert not attempt.validation_passed and not attempt.execution_completed
        assert not attempt.resumed_original_task
        assert calls == 5
    finally:
        env.close()
