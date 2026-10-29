"""Bounded, rule-based reactive recovery built on the existing Panda interfaces."""

import logging
import time
from datetime import datetime, timezone
from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from recovera.llm import (
    LLMProviderError,
    RecoveryAttemptSummary,
    RecoveryRequest,
    StrategyCandidate,
    StrategyProposal,
    StrategyProvider,
    new_request_id,
    validate_proposal,
)
from recovera.monitoring import (
    EventType,
    MonitoringEvent,
    TaskExpectation,
    TaskPhase,
)
from recovera.motion import (
    PandaJointPlanner,
    PlanStatus,
    TrajectoryValidator,
    execute_trajectory,
    solve_site_ik,
)

log = logging.getLogger("recovera.RECOVERY")
planner_log = logging.getLogger("recovera.PLANNER")
validation_log = logging.getLogger("recovera.VALIDATION")
execution_log = logging.getLogger("recovera.EXECUTION")
verify_log = logging.getLogger("recovera.VERIFY")


class RecoveryStrategy(str, Enum):
    REAPPROACH_AND_REGRASP = "reapproach_and_regrasp"
    CENTERED_REAPPROACH_AND_REGRASP = "centered_reapproach_and_regrasp"
    REFRESH_AND_REPLAN = "refresh_and_replan"
    SAFE_STOP = "safe_stop"


class RecoveryAttempt(BaseModel):
    model_config = ConfigDict(frozen=True)
    event_type: EventType
    event_step: int
    attempt_number: int
    strategy: RecoveryStrategy | None = None
    planning_status: PlanStatus | None = None
    planning_reason: str | None = None
    validation_passed: bool = False
    execution_completed: bool = False
    verification: str = "not_attempted"
    resumed_original_task: bool = False
    terminal_reason: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_proposal_accepted: bool | None = None
    llm_proposal_reason: str | None = None
    proposal_rationale: str | None = Field(default=None, max_length=400)
    llm_latency_seconds: float | None = None
    fallback_reason: str | None = None
    candidate_strategy_ids: tuple[str, ...] = ()
    candidate_feasibility: dict[str, str] = Field(default_factory=dict)
    inference_mode: str | None = None
    snapshot_step: int | None = None
    scene_revision: int | None = None
    recovery_duration_seconds: float | None = None
    trial_id: str | None = None
    scenario: str | None = None
    outcome_classification: str = "not_attempted"
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def classify_outcome(self):
        if self.resumed_original_task:
            classification = "recovery_success"
        elif self.terminal_reason in {
            "llm_abstained",
            "no_feasible_candidate",
            "invalid_state_or_unsupported_strategy",
            "llm_rejected_or_unavailable",
            "retry_limit_exhausted",
            "duplicate_event",
            "viewer_closed_during_recovery_setup",
            "reachability_safe_stop",
        }:
            classification = "safe_termination"
        elif self.terminal_reason:
            classification = "recovery_failure"
        else:
            classification = "not_attempted"
        object.__setattr__(self, "outcome_classification", classification)
        return self


class RecoveryPolicy:
    """Select only the operations supported by this Chunk 3 implementation."""

    def choose(
        self, event: MonitoringEvent, snapshot, task, expectation: TaskExpectation
    ) -> RecoveryStrategy | None:
        if not snapshot.valid or task.phase in (TaskPhase.COMPLETE, TaskPhase.INITIALIZING):
            return None
        if event.event_type in (EventType.FAILED_GRASP, EventType.OBJECT_DROPPED):
            if event.event_type is EventType.FAILED_GRASP and not expectation.grasp_attempted:
                return None
            if event.event_type is EventType.OBJECT_DROPPED and (
                event.entity_id != "cube"
                or event.expected_phase not in (TaskPhase.CARRYING, TaskPhase.PLACING)
                or not expectation.previously_held
            ):
                return None
            return RecoveryStrategy.CENTERED_REAPPROACH_AND_REGRASP
        if (
            event.event_type is EventType.SCENE_CHANGED
            and task.current_waypoint is not None
            and not expectation.previously_held
        ):
            return RecoveryStrategy.REFRESH_AND_REPLAN
        return None

    def candidates(
        self,
        event: MonitoringEvent,
        snapshot,
        task,
        expectation: TaskExpectation,
        *,
        excluded: set[str] | None = None,
    ) -> tuple[StrategyCandidate, ...]:
        """Return only action variants implemented by trusted coordinator code."""
        excluded = excluded or set()
        if not snapshot.valid or task.phase in (TaskPhase.COMPLETE, TaskPhase.INITIALIZING):
            return ()
        if (
            event.event_type is EventType.FAILED_GRASP
            and expectation.grasp_attempted
            and not expectation.previously_held
            and task.phase in (TaskPhase.GRASPING, TaskPhase.APPROACHING, TaskPhase.FAILED)
        ):
            candidates = (
                StrategyCandidate(
                    strategy_id=RecoveryStrategy.REAPPROACH_AND_REGRASP.value,
                    target_subgoal="reacquire_cube_calibrated_offset",
                    description=("Repeat the calibrated +8 mm X grasp approach that just failed."),
                    prerequisites=("grasp_attempted",),
                    requires_replanning=True,
                ),
                StrategyCandidate(
                    strategy_id=RecoveryStrategy.CENTERED_REAPPROACH_AND_REGRASP.value,
                    target_subgoal="reacquire_cube_centered",
                    description=(
                        "Shift the approach to zero X offset relative to the failed +8 mm attempt."
                    ),
                    prerequisites=("grasp_attempted",),
                    requires_replanning=True,
                ),
            )
        elif (
            event.event_type is EventType.OBJECT_DROPPED
            and event.entity_id == "cube"
            and event.expected_phase in (TaskPhase.CARRYING, TaskPhase.PLACING)
            and task.phase in (TaskPhase.CARRYING, TaskPhase.APPROACHING, TaskPhase.FAILED)
        ):
            candidates = (
                StrategyCandidate(
                    strategy_id=RecoveryStrategy.REAPPROACH_AND_REGRASP.value,
                    target_subgoal="reacquire_dropped_cube_calibrated_offset",
                    description="Approach the currently observed dropped-cube pose at +8 mm X.",
                    prerequisites=("confirmed_drop_event",),
                    requires_replanning=True,
                ),
                StrategyCandidate(
                    strategy_id=RecoveryStrategy.CENTERED_REAPPROACH_AND_REGRASP.value,
                    target_subgoal="reacquire_dropped_cube_centered",
                    description="Approach the currently observed dropped-cube pose with zero X offset.",
                    prerequisites=("confirmed_drop_event",),
                    requires_replanning=True,
                ),
                StrategyCandidate(
                    strategy_id=RecoveryStrategy.SAFE_STOP.value,
                    target_subgoal="leave_dropped_object_at_rest",
                    description="Do not attempt a grasp; leave the observed resting object in place.",
                    prerequisites=("confirmed_drop_event",),
                    requires_replanning=False,
                ),
            )
        elif event.event_type is EventType.SCENE_CHANGED:
            selected = self.choose(event, snapshot, task, expectation)
            if selected is None:
                return ()
            candidates = (
                StrategyCandidate(
                    strategy_id=selected.value,
                    target_subgoal="current_waypoint",
                    description="Replan to the current task waypoint through the observed scene.",
                    prerequisites=("current_waypoint_available",),
                    requires_replanning=True,
                ),
                StrategyCandidate(
                    strategy_id=RecoveryStrategy.SAFE_STOP.value,
                    target_subgoal="leave_object_at_rest",
                    description="Refuse the route and leave the resting object in place.",
                    prerequisites=("world_state_valid",),
                    requires_replanning=False,
                ),
            )
        else:
            selected = self.choose(event, snapshot, task, expectation)
            if selected is None:
                return ()
            subgoal = (
                "current_waypoint"
                if selected is RecoveryStrategy.REFRESH_AND_REPLAN
                else "reacquire_cube_calibrated_offset"
            )
            required = (
                "current_waypoint_available"
                if selected is RecoveryStrategy.REFRESH_AND_REPLAN
                else "object_previously_held"
                if event.event_type is EventType.OBJECT_DROPPED
                else "grasp_attempted"
            )
            candidates = (
                StrategyCandidate(
                    strategy_id=selected.value,
                    target_subgoal=subgoal,
                    description="Use the deterministic recovery route supported for this event.",
                    prerequisites=(required,),
                    requires_replanning=True,
                ),
            )
        return tuple(candidate for candidate in candidates if candidate.strategy_id not in excluded)


class RecoveryCoordinator:
    """Refresh, select, plan, validate, execute, and verify one bounded attempt."""

    def __init__(
        self,
        env,
        robot,
        *,
        max_attempts: int = 2,
        planner=None,
        policy=None,
        llm_provider: StrategyProvider | None = None,
        llm_enabled: bool = False,
        llm_fallback: str = "deterministic",
        llm_total_budget_seconds: float = 16.0,
    ):
        self.env, self.robot = env, robot
        self.max_attempts = max_attempts
        self.planner = planner or PandaJointPlanner(env, robot)
        self.validator = TrajectoryValidator(self.planner)
        self.policy = policy or RecoveryPolicy()
        self.llm_provider = llm_provider
        self.llm_enabled = llm_enabled
        self.llm_fallback = llm_fallback
        self.llm_total_budget_seconds = llm_total_budget_seconds
        self._llm_elapsed = 0.0
        self._seen: set[tuple[EventType, int]] = set()
        self._attempt_count = 0
        self._accepted_proposals: set[tuple[str, EventType, int, tuple[float, ...]]] = set()

    def reset(self) -> None:
        self._seen.clear()
        self._attempt_count = 0
        self._llm_elapsed = 0.0
        self._accepted_proposals.clear()
        reset_planner = getattr(self.planner, "reset", None)
        if reset_planner is not None:
            reset_planner()

    def recover(
        self,
        event: MonitoringEvent,
        task,
        *,
        allow_same_event_retry: bool = False,
        exclude_strategy_ids: set[str] | None = None,
    ) -> RecoveryAttempt:
        recovery_started = time.perf_counter()
        log.info("failure event received: %s at step=%d", event.event_type.value, event.step)
        key = (event.event_type, event.step)
        if key in self._seen and not allow_same_event_retry:
            return RecoveryAttempt(
                event_type=event.event_type,
                event_step=event.step,
                attempt_number=self._attempt_count,
                terminal_reason="duplicate_event",
            )
        self._seen.add(key)
        if self._attempt_count >= self.max_attempts:
            return RecoveryAttempt(
                event_type=event.event_type,
                event_step=event.step,
                attempt_number=self._attempt_count,
                terminal_reason="retry_limit_exhausted",
            )
        self._attempt_count += 1
        number = self._attempt_count

        # Always refresh from the live MuJoCo state; the event is a trigger only.
        if event.event_type in (EventType.FAILED_GRASP, EventType.OBJECT_DROPPED):
            self.robot.command_gripper(0.08)
            self.env.step(250)
            if self.env.viewer_closed:
                return RecoveryAttempt(
                    event_type=event.event_type,
                    event_step=event.step,
                    attempt_number=number,
                    terminal_reason="viewer_closed_during_recovery_setup",
                    trial_id=getattr(task, "trial_id", None),
                    scenario=getattr(task, "scenario", None),
                    recovery_duration_seconds=time.perf_counter() - recovery_started,
                )
        snapshot = task.monitor.extractor.capture()
        log.info(
            "refreshed world state: valid=%s sim_time=%.3fs cube=(%.3f, %.3f, %.3f)",
            snapshot.valid,
            snapshot.sim_time,
            *snapshot.cube_position,
        )
        expectation = TaskExpectation(
            phase=task.phase,
            active=True,
            grasp_attempted=task.grasp_attempted,
            grasp_confirmed=task.grasp_confirmed,
            previously_held=task.previously_held,
        )
        strategy = self.policy.choose(event, snapshot, task, expectation)
        candidates = self.policy.candidates(
            event,
            snapshot,
            task,
            expectation,
            excluded=exclude_strategy_ids,
        )
        allowed_ids = {candidate.strategy_id for candidate in candidates}
        if strategy is not None and strategy.value not in allowed_ids:
            strategy = RecoveryStrategy(candidates[0].strategy_id) if candidates else None
        candidate_plans = {}
        feasibility = {}
        feasibility_status = {}
        for candidate in candidates:
            if candidate.strategy_id == RecoveryStrategy.SAFE_STOP.value:
                feasibility[candidate.strategy_id] = "safe stop is always permitted; no robot motion"
                candidate_plans[candidate.strategy_id] = {"safe_stop": True}
                continue
            bundle, reason, status = self._preflight_candidate(candidate, snapshot, task)
            feasibility[candidate.strategy_id] = reason
            feasibility_status[candidate.strategy_id] = status
            if bundle is not None:
                candidate_plans[candidate.strategy_id] = bundle
            log.info(
                "candidate feasibility strategy=%s result=%s reason=%s",
                candidate.strategy_id,
                "feasible" if bundle is not None else "rejected",
                reason,
            )
        candidates = tuple(
            candidate for candidate in candidates if candidate.strategy_id in candidate_plans
        )
        log.info(
            "feasible recovery candidates event=%s candidates=%s",
            event.event_type.value,
            tuple(candidate.strategy_id for candidate in candidates),
        )
        llm_trace = {
            "provider": None,
            "model": None,
            "accepted": None,
            "reason": None,
            "latency": None,
            "fallback": None,
            "inference_mode": getattr(self.llm_provider, "inference_mode", None),
            "candidates": tuple(candidate.strategy_id for candidate in candidates),
            "snapshot_step": snapshot.step,
            "scene_revision": snapshot.scene_revision,
            "trial_id": getattr(task, "trial_id", None),
            "scenario": getattr(task, "scenario", None),
            "started": recovery_started,
            "abstain": False,
            "feasibility": feasibility,
        }
        if not candidates:
            summary = "; ".join(f"{key}: {value}" for key, value in feasibility.items())
            log.warning("recovery stopped: no feasible candidate (%s)", summary or "none generated")
            statuses = [status for status in feasibility_status.values() if status is not None]
            return RecoveryAttempt(
                event_type=event.event_type,
                event_step=event.step,
                attempt_number=number,
                planning_status=statuses[0] if statuses else None,
                planning_reason=summary or "policy generated no candidate",
                terminal_reason="no_feasible_candidate" if feasibility else "invalid_state_or_unsupported_strategy",
                candidate_strategy_ids=(),
                candidate_feasibility=feasibility,
                snapshot_step=snapshot.step,
                scene_revision=snapshot.scene_revision,
                recovery_duration_seconds=time.perf_counter() - recovery_started,
                trial_id=llm_trace["trial_id"],
                scenario=llm_trace["scenario"],
            )
        allowed_ids = {candidate.strategy_id for candidate in candidates}
        if strategy is None or strategy.value not in allowed_ids:
            strategy = RecoveryStrategy(candidates[0].strategy_id)
        llm_trace["candidates"] = tuple(candidate.strategy_id for candidate in candidates)
        if self.llm_enabled and candidates:
            strategy, snapshot, expectation, llm_trace = self._llm_select(
                event, snapshot, task, expectation, strategy, candidates, feasibility
            )
            llm_trace["feasibility"] = feasibility
        elif (
            exclude_strategy_ids and strategy is not None and strategy.value in exclude_strategy_ids
        ):
            strategy = next(
                (
                    RecoveryStrategy(candidate.strategy_id)
                    for candidate in candidates
                    if candidate.strategy_id not in exclude_strategy_ids
                ),
                None,
            )
        if not snapshot.valid or strategy is None:
            reason = llm_trace["fallback"] or "invalid state or no supported strategy"
            log.warning("recovery refused: %s", reason)
            return RecoveryAttempt(
                event_type=event.event_type,
                event_step=event.step,
                attempt_number=number,
                terminal_reason="llm_abstained"
                if llm_trace.get("abstain")
                else "llm_rejected_or_unavailable"
                if self.llm_enabled
                else "invalid_state_or_unsupported_strategy",
                llm_provider=llm_trace["provider"],
                llm_model=llm_trace["model"],
                llm_proposal_accepted=llm_trace["accepted"],
                llm_proposal_reason=llm_trace["reason"],
                llm_latency_seconds=llm_trace["latency"],
                fallback_reason=llm_trace["fallback"],
                candidate_strategy_ids=llm_trace["candidates"],
                candidate_feasibility=llm_trace.get("feasibility", {}),
                inference_mode=llm_trace["inference_mode"],
                snapshot_step=snapshot.step,
                scene_revision=snapshot.scene_revision,
                recovery_duration_seconds=time.perf_counter() - recovery_started,
                trial_id=llm_trace["trial_id"],
                scenario=llm_trace["scenario"],
            )
        log.info("selected strategy: %s", strategy.value)
        if strategy is RecoveryStrategy.SAFE_STOP:
            return RecoveryAttempt(
                event_type=event.event_type,
                event_step=event.step,
                attempt_number=number,
                strategy=strategy,
                planning_reason="all grasp candidates failed bounded IK/collision preflight; object left resting",
                terminal_reason="reachability_safe_stop",
                **self._llm_result_fields(llm_trace),
            )
        planner_log.debug(
            "recovery joint request start=%s goal_context=%s",
            np.round(self.robot.joint_positions, 4).tolist(),
            np.round(np.asarray(snapshot.cube_position), 4).tolist()
            if strategy is RecoveryStrategy.REAPPROACH_AND_REGRASP
            else task.current_waypoint,
        )
        if strategy in (
            RecoveryStrategy.REAPPROACH_AND_REGRASP,
            RecoveryStrategy.CENTERED_REAPPROACH_AND_REGRASP,
        ):
            task._set_phase(TaskPhase.APPROACHING)

        bundle = candidate_plans[strategy.value]
        plan = bundle["approach_plan"] if "approach_plan" in bundle else bundle["main_plan"]
        validation = (
            bundle["approach_validation"]
            if "approach_validation" in bundle
            else bundle["main_validation"]
        )
        validation_log.info(
            "selected trajectory preflight %s: %s",
            "accepted" if validation.valid else "rejected",
            validation.reason,
        )
        validation = self.validator.validate(plan, self.robot.joint_positions)
        if not validation.valid:
            return RecoveryAttempt(
                event_type=event.event_type,
                event_step=event.step,
                attempt_number=number,
                strategy=strategy,
                planning_status=plan.status,
                planning_reason=f"selected approach revalidation failed: {validation.reason}",
                terminal_reason="validation_failed",
                **self._llm_result_fields(llm_trace),
            )
        execution_log.info("executing approved trajectory with Panda position servos")
        moved, execution_reason = execute_trajectory(self.robot, plan.trajectory)
        execution_log.info("execution %s: %s", "completed" if moved else "failed", execution_reason)
        if not moved:
            return RecoveryAttempt(
                event_type=event.event_type,
                event_step=event.step,
                attempt_number=number,
                strategy=strategy,
                planning_status=plan.status,
                planning_reason=execution_reason,
                validation_passed=True,
                terminal_reason="execution_failed",
                **self._llm_result_fields(llm_trace),
            )
        if strategy is RecoveryStrategy.REFRESH_AND_REPLAN:
            reached = (
                np.linalg.norm(self.robot.end_effector_position - bundle["target_position"]) <= 0.01
            )
            verify_log.info(
                "current waypoint verification: %s", "reached" if reached else "not reached"
            )
            return RecoveryAttempt(
                event_type=event.event_type,
                event_step=event.step,
                attempt_number=number,
                strategy=strategy,
                planning_status=plan.status,
                planning_reason=f"{plan.reason}; {validation.reason}",
                validation_passed=True,
                execution_completed=True,
                verification="current waypoint reached" if reached else "waypoint_not_reached",
                resumed_original_task=bool(reached),
                terminal_reason=None if reached else "replan_verification_failed",
                **self._llm_result_fields(llm_trace),
            )

        # Reacquisition is a distinct subgoal: descend, close, maintain contact,
        # then verify bilateral pads and cube stability before resuming placement.
        task._set_phase(TaskPhase.APPROACHING)
        task.env.set_robot_cube_contacts(True)
        descend = bundle["descend_target"]
        descend_plan = bundle["descend_plan"]
        descent_validation = self.validator.validate(descend_plan, self.robot.joint_positions)
        validation_log.info(
            "descent trajectory %s: %s",
            "accepted" if descent_validation.valid else "rejected",
            descent_validation.reason,
        )
        if not descent_validation.valid:
            return RecoveryAttempt(
                event_type=event.event_type,
                event_step=event.step,
                attempt_number=number,
                strategy=strategy,
                planning_status=descend_plan.status,
                planning_reason=f"descent validation failed: {descent_validation.reason}",
                validation_passed=False,
                execution_completed=False,
                terminal_reason="validation_failed",
                **self._llm_result_fields(llm_trace),
            )
        execution_log.info("executing validated descent trajectory")
        reached, descent_reason = execute_trajectory(self.robot, descend_plan.trajectory)
        pose_reached = (
            reached and np.linalg.norm(self.robot.end_effector_position - descend) <= 0.01
        )
        execution_log.info(
            "descent execution %s: %s; endpoint_pose_verified=%s",
            "completed" if reached else "failed",
            descent_reason,
            pose_reached,
        )
        if not pose_reached:
            task._set_phase(TaskPhase.FAILED)
            return RecoveryAttempt(
                event_type=event.event_type,
                event_step=event.step,
                attempt_number=number,
                strategy=strategy,
                planning_status=descend_plan.status,
                planning_reason="planned descent endpoint did not reach the grasp pose",
                validation_passed=True,
                execution_completed=bool(reached),
                verification="grasp_pose_not_reached",
                terminal_reason="recovery_pose_not_reached",
                **self._llm_result_fields(llm_trace),
            )
        task.monitor.begin_grasp_attempt()
        task._set_phase(TaskPhase.GRASPING)
        task.grasp_attempted = True
        task.grasp_confirmed = False
        self.robot.command_gripper(0.030)
        verify_log.info("checking bilateral grasp and cube stability after reapproach")
        stable = False
        if pose_reached:
            for _ in range(600):
                self.env.step()
                if self.robot.has_grasp_contact():
                    self.robot.hold_gripper(preload=0.020)
                    for _ in range(50):
                        self.env.step()
                    displacement = np.linalg.norm(
                        self.robot.cube_position[:2] - np.asarray(snapshot.cube_position)[:2]
                    )
                    stable = self.robot.has_grasp_contact() and displacement <= 0.010
                    break
        task.grasp_confirmed = stable
        task.previously_held = stable
        if stable:
            task._set_phase(TaskPhase.CARRYING)
        else:
            task._set_phase(TaskPhase.FAILED)
        verify_log.info(
            "regrasp verification=%s bilateral_contact=%s",
            "confirmed" if stable else "unverified",
            self.robot.has_grasp_contact(),
        )
        return RecoveryAttempt(
            event_type=event.event_type,
            event_step=event.step,
            attempt_number=number,
            strategy=strategy,
            planning_status=plan.status,
            planning_reason=validation.reason,
            validation_passed=True,
            execution_completed=True,
            verification="bilateral grasp confirmed" if stable else "grasp_not_confirmed",
            resumed_original_task=stable,
            terminal_reason=None if stable else "regrasp_verification_failed",
            **self._llm_result_fields(llm_trace),
        )

    @staticmethod
    def _llm_result_fields(trace: dict) -> dict:
        return {
            "llm_provider": trace["provider"],
            "llm_model": trace["model"],
            "llm_proposal_accepted": trace["accepted"],
            "llm_proposal_reason": trace["reason"],
            "proposal_rationale": trace.get("rationale"),
            "llm_latency_seconds": trace["latency"],
            "fallback_reason": trace["fallback"],
            "candidate_strategy_ids": trace["candidates"],
            "candidate_feasibility": trace.get("feasibility", {}),
            "inference_mode": trace["inference_mode"],
            "snapshot_step": trace["snapshot_step"],
            "scene_revision": trace["scene_revision"],
            "timestamp_utc": datetime.now(timezone.utc),
            "recovery_duration_seconds": time.perf_counter() - trace["started"],
            "trial_id": trace["trial_id"],
            "scenario": trace["scenario"],
        }

    def _preflight_candidate(self, candidate, snapshot, task):
        """Plan and validate every stage before a candidate is offered to inference."""
        strategy = RecoveryStrategy(candidate.strategy_id)
        rotation = np.diag([-1.0, 1.0, -1.0])
        if strategy is RecoveryStrategy.REFRESH_AND_REPLAN:
            target = np.asarray(task.current_waypoint, dtype=float)
            goals = [solve_site_ik(self.robot, target, rotation)]
            if self.env.config.scenario_id == "obstacle-replan":
                # Search a bounded, seeded set of IK postures. This gives the
                # planner alternative end configurations around scene obstacles.
                rng = np.random.default_rng(task.config.random_seed)
                goals.extend(
                    solve_site_ik(
                        self.robot,
                        target,
                        rotation,
                        initial_joint_state=rng.uniform(self.planner.lower, self.planner.upper),
                    )
                    for _ in range(24)
                )
            plan = None
            goal = None
            last_reason = "IK failed for current waypoint"
            selected_posture_index = 0
            for posture_index, candidate_goal in enumerate(goals):
                if candidate_goal is None:
                    continue
                candidate_plan = self.planner.plan(
                    self.robot.joint_positions,
                    candidate_goal,
                    scene_revision=snapshot.scene_revision,
                )
                if candidate_plan.status is PlanStatus.SUCCESS:
                    goal, plan = candidate_goal, candidate_plan
                    selected_posture_index = posture_index
                    if posture_index:
                        log.info(
                            "scene-aware IK replan selected alternate posture index=%d after earlier posture rejection=%s",
                            posture_index,
                            last_reason,
                        )
                    break
                last_reason = candidate_plan.reason
            if plan is None or goal is None:
                return None, last_reason, PlanStatus.NO_PATH
            validation = self.validator.validate(plan, self.robot.joint_positions)
            if not validation.valid:
                return None, validation.reason, plan.status
            return (
                {
                    "main_plan": plan,
                    "main_validation": validation,
                    "target_position": target,
                },
                (
                    f"{plan.reason}; alternate IK posture index={selected_posture_index}; "
                    f"prior posture result={last_reason}; {validation.reason}"
                    if selected_posture_index
                    else f"{plan.reason}; {validation.reason}"
                ),
                plan.status,
            )

        x_offset = 0.0 if strategy is RecoveryStrategy.CENTERED_REAPPROACH_AND_REGRASP else 0.008
        self.env.set_robot_cube_contacts(True)
        approach_target = np.asarray(snapshot.cube_position) + [x_offset, 0.0, 0.1766]
        approach_goal = solve_site_ik(self.robot, approach_target, rotation)
        if approach_goal is None:
            return None, "IK failed for pregrasp hover pose", PlanStatus.NO_PATH
        approach_plan = self.planner.plan(
            self.robot.joint_positions,
            approach_goal,
            scene_revision=snapshot.scene_revision,
        )
        if approach_plan.status is not PlanStatus.SUCCESS:
            return None, f"pregrasp plan failed: {approach_plan.reason}", approach_plan.status
        approach_validation = self.validator.validate(approach_plan, self.robot.joint_positions)
        if not approach_validation.valid:
            return (
                None,
                f"pregrasp validation failed: {approach_validation.reason}",
                approach_plan.status,
            )

        descend_target = np.asarray(snapshot.cube_position) + [x_offset, 0.0, 0.0066]
        descend_goal = solve_site_ik(
            self.robot,
            descend_target,
            rotation,
            initial_joint_state=approach_goal,
        )
        if descend_goal is None:
            return None, "IK failed for grasp pose", PlanStatus.NO_PATH
        descend_plan = self.planner.plan(
            approach_goal,
            descend_goal,
            scene_revision=snapshot.scene_revision,
            allow_initial_finger_contact=False,
            allow_goal_finger_contact=True,
        )
        if descend_plan.status is not PlanStatus.SUCCESS:
            return None, f"grasp descent plan failed: {descend_plan.reason}", descend_plan.status
        descend_validation = self.validator.validate(descend_plan, approach_goal)
        if not descend_validation.valid:
            return (
                None,
                f"grasp descent validation failed: {descend_validation.reason}",
                descend_plan.status,
            )
        return (
            {
                "approach_plan": approach_plan,
                "approach_validation": approach_validation,
                "approach_target": approach_target,
                "descend_plan": descend_plan,
                "descend_validation": descend_validation,
                "descend_target": descend_target,
                "rotation": rotation,
            },
            "pregrasp and grasp descent plans passed collision/limit validation",
            PlanStatus.SUCCESS,
        )

    def _llm_select(self, event, snapshot, task, expectation, deterministic, candidates=None, feasibility=None):
        if candidates is None:
            candidates = self.policy.candidates(event, snapshot, task, expectation)
        provider = self.llm_provider
        trace = {
            "provider": getattr(provider, "provider_name", None),
            "model": getattr(provider, "model_name", "unknown"),
            "accepted": False,
            "reason": None,
            "latency": None,
            "fallback": None,
            "inference_mode": getattr(provider, "inference_mode", None),
            "candidates": tuple(candidate.strategy_id for candidate in candidates),
            "snapshot_step": snapshot.step,
            "scene_revision": snapshot.scene_revision,
            "trial_id": getattr(task, "trial_id", None),
            "scenario": getattr(task, "scenario", None),
            "started": time.perf_counter(),
            "abstain": False,
            "feasibility": feasibility or {},
            "rationale": None,
        }
        fallback_strategy = (
            deterministic
            if self.llm_fallback == "deterministic"
            and deterministic is not None
            and deterministic.value in trace["candidates"]
            else None
        )
        if provider is None:
            trace["reason"] = "no LLM provider is configured"
            trace["fallback"] = trace["reason"]
            log.warning("LLM guidance unavailable; fallback=%s", trace["fallback"])
            return fallback_strategy, snapshot, expectation, trace
        remaining = self.llm_total_budget_seconds - self._llm_elapsed
        if remaining <= 0:
            trace["reason"] = "per-trial LLM time budget exhausted"
            trace["fallback"] = trace["reason"]
            return fallback_strategy, snapshot, expectation, trace
        request = self._build_request(event, snapshot, task, expectation, candidates, trace.get("feasibility", {}))
        timeout = min(float(getattr(provider, "timeout_seconds", remaining)), remaining)
        log.info(
            "LLM request started mode=%s provider=%s model=%s candidates=%s",
            trace["inference_mode"],
            trace["provider"],
            trace["model"],
            trace["candidates"],
        )
        started = time.perf_counter()
        try:
            raw_proposal = provider.propose(request, timeout_seconds=timeout)
            proposal = StrategyProposal.model_validate(raw_proposal)
        except (LLMProviderError, TimeoutError, ValidationError) as exc:
            latency = time.perf_counter() - started
            self._llm_elapsed += latency
            trace["latency"] = latency
            trace["reason"] = f"{type(exc).__name__}: {exc}"
            trace["fallback"] = trace["reason"]
            log.warning(
                "LLM request failed provider=%s model=%s latency=%.3fs; fallback=%s",
                trace["provider"],
                trace["model"],
                latency,
                trace["fallback"],
            )
            return fallback_strategy, snapshot, expectation, trace
        latency = time.perf_counter() - started
        self._llm_elapsed += latency
        trace["latency"] = latency
        log.info(
            "LLM request completed provider=%s model=%s latency=%.3fs",
            trace["provider"],
            trace["model"],
            latency,
        )

        latest = task.monitor.extractor.capture()
        if not self._same_observation(snapshot, latest):
            trace["reason"] = "proposal rejected: world state became stale during request"
            trace["fallback"] = "world state changed while awaiting local model"
            return None, latest, expectation, trace

        validation = validate_proposal(request, proposal)
        trace["reason"] = validation.reason
        if validation.accepted and not proposal.abstain:
            trace["rationale"] = proposal.rationale
        if validation.accepted and proposal.abstain:
            trace["abstain"] = True
            trace["fallback"] = "model explicitly abstained; safe stop"
            log.warning("LLM explicitly abstained; recovery stops safely")
            return None, snapshot, expectation, trace
        if validation.accepted:
            signature = (
                proposal.strategy_id,
                event.event_type,
                snapshot.scene_revision,
                tuple(round(value, 3) for value in snapshot.cube_position),
            )
            if signature in self._accepted_proposals:
                validation = validation.model_copy(
                    update={
                        "accepted": False,
                        "reason": "repeated proposal for unchanged event state",
                    }
                )
                trace["reason"] = validation.reason
            else:
                self._accepted_proposals.add(signature)
        if validation.accepted:
            trace["accepted"] = True
            log.info("LLM proposal accepted strategy=%s", proposal.strategy_id)
            selected = RecoveryStrategy(proposal.strategy_id)
            log.info(
                "LLM selected candidate=%s (candidate_count=%d)", selected.value, len(candidates)
            )
            return selected, snapshot, expectation, trace
        log.warning("LLM proposal rejected: %s", validation.reason)
        trace["fallback"] = validation.reason
        return fallback_strategy, snapshot, expectation, trace

    def _build_request(self, event, snapshot, task, expectation, candidates, candidate_feasibility=None):
        prerequisites = {
            "world_state_valid": bool(snapshot.valid),
            "grasp_attempted": expectation.grasp_attempted,
            "object_previously_held": expectation.previously_held,
            "confirmed_drop_event": event.event_type is EventType.OBJECT_DROPPED,
            "current_waypoint_available": task.current_waypoint is not None,
        }
        attempts = tuple(
            RecoveryAttemptSummary(
                event_type=attempt.event_type,
                strategy_id=attempt.strategy.value if attempt.strategy else None,
                verification=attempt.verification,
                terminal_reason=attempt.terminal_reason,
            )
            for attempt in task.recovery_attempts[-5:]
        )
        scene = getattr(self.env, "last_scene_change", None)
        return RecoveryRequest(
            request_id=new_request_id(),
            task_phase=task.phase,
            original_task_goal_xyz=tuple(float(v) for v in task.config.target_position),
            last_task_action=(
                "top_down_grasp_at_calibrated_plus_8mm_x_offset"
                if event.event_type is EventType.FAILED_GRASP
                else None
            ),
            last_task_action_outcome=(
                "confirmed_no_bilateral_fingertip_contact"
                if event.event_type is EventType.FAILED_GRASP
                else None
            ),
            failure={
                "event_type": event.event_type,
                "step": event.step,
                "sim_time": event.sim_time,
                "entity_id": event.entity_id,
                "explanation": event.explanation[:240],
            },
            observed_step=snapshot.step,
            observed_sim_time=snapshot.sim_time,
            cube_position_xyz=snapshot.cube_position,
            cube_orientation_wxyz=snapshot.cube_orientation,
            bilateral_finger_contact=snapshot.bilateral_finger_contact,
            robot_joint_positions_rad=tuple(float(v) for v in self.robot.joint_positions),
            scene_revision=snapshot.scene_revision,
            last_scene_change=f"{scene[0]}:{scene[1]}" if scene else None,
            state_valid=snapshot.valid,
            prerequisites=prerequisites,
            candidates=candidates,
            candidate_feasibility=candidate_feasibility or {},
            previous_attempts=attempts,
        )

    @staticmethod
    def _same_observation(before, after) -> bool:
        return (
            after.valid
            and before.step == after.step
            and before.scene_revision == after.scene_revision
            and np.allclose(before.cube_position, after.cube_position, atol=1e-8)
            and np.allclose(before.cube_orientation, after.cube_orientation, atol=1e-8)
            and np.allclose(before.cube_linear_velocity, after.cube_linear_velocity, atol=1e-8)
            and np.allclose(before.cube_angular_velocity, after.cube_angular_velocity, atol=1e-8)
            and np.allclose(before.gripper_position, after.gripper_position, atol=1e-8)
            and before.bilateral_finger_contact == after.bilateral_finger_contact
        )
