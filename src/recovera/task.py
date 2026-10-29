"""Deterministic baseline pick-and-place sequence."""

import logging
import time

import numpy as np

from recovera.config import RecoveraConfig
from recovera.environment import MujocoEnvironment
from recovera.llm import MockStrategyProvider, OllamaProvider
from recovera.monitoring import (
    EventType,
    MonitoringEvent,
    TaskExpectation,
    TaskPhase,
    WorldStateMonitor,
)
from recovera.motion import (
    PandaJointPlanner,
    PlanStatus,
    TrajectoryValidator,
    execute_trajectory,
    solve_site_ik,
)
from recovera.recovery import RecoveryAttempt, RecoveryCoordinator
from recovera.results import TrialResult, TrialStatus
from recovera.robot import PandaAdapter
from recovera.scenarios import get_scenario
from recovera.verification import verify_stable

log = logging.getLogger("recovera.SIMULATION")
monitor_log = logging.getLogger("recovera.MONITOR")


class PickAndPlaceTask:
    """Execute Cartesian approach, grasp, lift, transport, release and verification."""

    def __init__(
        self,
        env: MujocoEnvironment,
        robot: PandaAdapter,
        config: RecoveraConfig,
        *,
        trial_id: str | None = None,
    ):
        self.env, self.robot, self.config = env, robot, config
        self.trial_id = trial_id
        self.scenario = config.scenario_id if config.scenario_id != "baseline" else (
            "recovery-demo"
            if config.demo_fail_first_grasp
            else "drop-demo"
            if config.demo_drop_after_grasp
            else "baseline"
        )
        self.monitor = WorldStateMonitor(env)
        self.phase = TaskPhase.INITIALIZING
        self.grasp_attempted = False
        self.grasp_confirmed = False
        self.previously_held = False
        self.current_waypoint: np.ndarray | None = None
        llm_provider = None
        if config.llm_enabled and config.llm_provider == "ollama":
            llm_provider = OllamaProvider(
                endpoint=config.llm_endpoint,
                model=config.llm_model,
                timeout_seconds=config.llm_timeout_seconds,
                temperature=config.llm_temperature,
                num_predict=config.llm_num_predict,
            )
        elif config.llm_enabled and config.llm_provider == "mock":
            llm_provider = MockStrategyProvider(config.llm_mock_strategy)
        self.llm_provider = llm_provider
        self.recovery = RecoveryCoordinator(
            env,
            robot,
            max_attempts=config.max_recovery_attempts,
            llm_provider=llm_provider,
            llm_enabled=config.llm_enabled,
            llm_fallback=config.llm_fallback,
            llm_total_budget_seconds=config.llm_total_budget_seconds,
            planner=PandaJointPlanner(env, robot, seed=config.random_seed),
        )
        self.recovery_attempts: list[RecoveryAttempt] = []
        self.env.add_step_observer(self._monitor_step)
        log.info("Panda adapter, task, monitor, and recovery coordinator initialized")
        log.info(
            "LLM guidance enabled=%s provider=%s inference_mode=%s model=%s fallback=%s",
            config.llm_enabled,
            config.llm_provider if config.llm_enabled else "disabled",
            getattr(llm_provider, "inference_mode", "none"),
            getattr(llm_provider, "model_name", "n/a"),
            config.llm_fallback,
        )
        log.info(
            "initial state cube=(%.3f, %.3f, %.3f) joint_norm=%.3f",
            *self.robot.cube_position,
            float(np.linalg.norm(self.robot.joint_positions)),
        )

    def _set_phase(self, phase: TaskPhase) -> None:
        if phase is self.phase:
            return
        self.phase = phase
        log.info("t=%.3fs task phase=%s", self.env.data.time, phase.value.upper())

    def _monitor_step(self) -> None:
        events = self.monitor.observe(
            TaskExpectation(
                phase=self.phase,
                active=self.phase not in (TaskPhase.COMPLETE, TaskPhase.FAILED),
                grasp_attempted=self.grasp_attempted,
                grasp_confirmed=self.grasp_confirmed,
                previously_held=self.previously_held,
            )
        )
        for event in events:
            monitor_log.warning(
                "t=%.3fs step=%d event=%s entity=%s reason=%s",
                event.sim_time,
                event.step,
                event.event_type.value.upper(),
                event.entity_id,
                event.explanation,
            )
        if any(event.event_type.value == "object_dropped" for event in events):
            if self.scenario == "beyond-reach":
                self.env.data.xfrc_applied[self.env.model.body("cube").id] = 0.0
            self._set_phase(TaskPhase.FAILED)

    def run(self) -> TrialResult:
        started = time.monotonic()
        self.env.reset()
        self.monitor.reset()
        self.recovery.reset()
        self.recovery_attempts.clear()
        self.current_waypoint = None
        self._set_phase(TaskPhase.INITIALIZING)
        self.grasp_attempted = self.grasp_confirmed = self.previously_held = False
        if not self.robot.initialize():
            return self._result(
                started, False, "Panda gripper failed its open-state initialization"
            )
        if self.scenario == "blocked-route":
            target = np.asarray(self.config.cube_initial_position) + [0.008, 0.0, 0.1766]
            goal = solve_site_ik(self.robot, target, np.diag([-1.0, 1.0, -1.0]))
            planner = PandaJointPlanner(self.env, self.robot, seed=self.config.random_seed)
            plan = None if goal is None else planner.plan(self.robot.joint_positions, goal)
            if goal is None or plan is None or plan.status is not PlanStatus.SUCCESS:
                reason = "obstacle-aware preflight confirmed no collision-free approach; cube left at rest"
                log.warning("scenario=blocked-route safe stop: %s", reason)
                self.current_waypoint = target.copy()
                self._set_phase(TaskPhase.APPROACHING)
                revision = self.env.record_scene_change("route_obstacle", "add")
                self.monitor._scene_revision = revision
                snapshot = self.monitor.extractor.capture()
                event = MonitoringEvent(
                    event_type=EventType.SCENE_CHANGED,
                    step=snapshot.step,
                    sim_time=snapshot.sim_time,
                    entity_id="route_obstacle",
                    expected_phase=TaskPhase.APPROACHING,
                    explanation="Configured static obstacle blocks the measured approach route.",
                    evidence={"preflight_plan_status": plan.status.value if plan else "ik_failed"},
                )
                self.monitor.record_event(event)
                self.recovery_attempts.append(self.recovery.recover(event, self))
                return self._result(started, False, reason, status=TrialStatus.SAFE_STOP, classification="safe_stop")
            return self._result(started, False, "infeasible-route fixture unexpectedly admitted a path", status=TrialStatus.INCOMPLETE, classification="incomplete")
        cube = self.robot.cube_position
        target = np.asarray(self.config.target_position)
        # The local fingertip direction must point down, while the finger
        # closing axis stays horizontal. Position-only IK let the Panda reach
        # the block with a tilted wrist, which pressed it into the table.
        grasp_rotation = np.diag([-1.0, 1.0, -1.0])
        # The model's wrist reaches its joint-7 limit in this pose. This fixed
        # site offset compensates the measured residual so the pads center on
        # the 24 mm block after IK converges.
        grasp_site = cube + np.array([0.008, 0.0, 0.0066])
        self._set_phase(TaskPhase.APPROACHING)
        approach_waypoints = (
            grasp_site + [0, 0, 0.17],
            grasp_site,
        )
        skip_first_approach = False
        if self.scenario == "obstacle-replan":
            self.current_waypoint = np.asarray(approach_waypoints[0], dtype=float).copy()
            revision = self.env.record_scene_change("route_obstacle", "add")
            # The configured obstacle is part of the compiled scene. Synchronize
            # the observer because this scenario event is published explicitly.
            self.monitor._scene_revision = revision
            snapshot = self.monitor.extractor.capture()
            event = MonitoringEvent(
                event_type=EventType.SCENE_CHANGED,
                step=snapshot.step,
                sim_time=snapshot.sim_time,
                entity_id="route_obstacle",
                expected_phase=TaskPhase.APPROACHING,
                explanation="The route obstacle is present in MuJoCo and planner collision geometry.",
                evidence={"scene_revision": revision},
            )
            self.monitor.record_event(event)
            attempt = self.recovery.recover(event, self)
            self.recovery_attempts.append(attempt)
            if not attempt.resumed_original_task:
                self._set_phase(TaskPhase.FAILED)
                return self._result(started, False, "obstacle-aware replan did not verify the approach")
            skip_first_approach = True
        for index, point in enumerate(approach_waypoints):
            if skip_first_approach and index == 0:
                continue
            if self.scenario == "obstacle-replan":
                moved = self._move_waypoint(point, grasp_rotation)
            else:
                moved = self.robot.move_to(
                    point,
                    self.config.max_steps_per_waypoint,
                    self.config.position_tolerance,
                    target_rotation=grasp_rotation,
                    max_joint_step=0.035,
                )
            if not moved:
                return self._result(started, False, "arm failed to reach cube approach")
        # Stop at a narrow grasp aperture so the position actuator maintains
        # contact instead of driving the fingers through their minimum gap.
        # The named demo suppresses contact for the first attempt only. The real
        # recovery coordinator restores contact before its regrasp attempt.
        self.env.set_robot_cube_contacts(not self.config.demo_fail_first_grasp)
        if self.config.demo_fail_first_grasp:
            logging.getLogger("recovera.SIMULATION").warning(
                "controlled demo: robot-cube contact disabled for initial grasp attempt"
            )
        self._set_phase(TaskPhase.GRASPING)
        self.grasp_attempted = True
        self.robot.command_gripper(0.030)
        grasped = False
        for _ in range(1500):
            self.robot.track_pose_step(grasp_site, grasp_rotation)
            self.env.step()
            if self.env.viewer_closed:
                self._set_phase(TaskPhase.FAILED)
                return self._result(started, False, "viewer closed during grasp attempt")
            if self.robot.has_grasp_contact():
                self.robot.hold_gripper(preload=0.020)
                for _ in range(50):
                    self.robot.track_pose_step(grasp_site, grasp_rotation)
                    self.env.step()
                    if self.env.viewer_closed:
                        self._set_phase(TaskPhase.FAILED)
                        return self._result(
                            started, False, "viewer closed during grasp verification"
                        )
                    if not self.robot.has_grasp_contact():
                        break
                displacement = np.linalg.norm(self.robot.cube_position[:2] - cube[:2])
                grasped = self.robot.has_grasp_contact() and displacement <= 0.004
                break
        if not grasped:
            event = next(
                (
                    item
                    for item in reversed(self.monitor.events)
                    if item.event_type.value == "failed_grasp"
                ),
                None,
            )
            grasped = self._recover_with_alternatives(event) if event is not None else False
            if not grasped:
                self._set_phase(TaskPhase.FAILED)
                return self._result(started, False, "gripper did not establish a verified grasp")
            cube = self.robot.cube_position
            grasp_site = cube + np.array([0.008, 0.0, 0.0066])
        self.grasp_confirmed = True
        self.previously_held = True
        if self.scenario == "obstacle-replan":
            self.robot.hold_gripper(preload=0.030)
        self._set_phase(TaskPhase.CARRYING)
        if self.config.demo_drop_after_grasp or self.scenario == "beyond-reach":
            logging.getLogger("recovera.SIMULATION").warning(
                "controlled scenario: injecting external horizontal force and releasing during lift"
            )
            if self.scenario == "beyond-reach":
                partial_lift = grasp_site + [0, 0, 0.06]
                if not self.robot.move_to(
                    partial_lift,
                    self.config.max_steps_per_waypoint,
                    self.config.position_tolerance,
                    target_rotation=grasp_rotation,
                    maintain_grasp=True,
                ):
                    self._set_phase(TaskPhase.FAILED)
                    return self._result(started, False, "arm failed before configured mid-air release")
                cube_body = self.env.model.body("cube").id
                self.env.data.xfrc_applied[cube_body, 0] = self.config.drop_force_newtons
            self.robot.command_gripper(0.08)
        place_site_z = target[2] + self.config.cube_half_size + 0.0066
        lift_height = 0.43 if self.scenario == "obstacle-replan" else 0.203
        transfer_height = max(0.48, place_site_z + 0.25) if self.scenario == "obstacle-replan" else max(0.20, place_site_z + 0.17)
        waypoints = (
            grasp_site + [0, 0, lift_height],
            np.array([target[0] + 0.008, target[1], transfer_height]),
            np.array([target[0] + 0.008, target[1], place_site_z]),
        )
        waypoint_names = ("lift", "transport", "place approach")
        for index, (name, point) in enumerate(zip(waypoint_names, waypoints)):
            self.current_waypoint = np.asarray(point, dtype=float).copy()
            if name == "place approach":
                self._set_phase(TaskPhase.PLACING)
            if not self._move_waypoint(point, grasp_rotation, maintain_grasp=True):
                self._set_phase(TaskPhase.FAILED)
                return self._result(started, False, f"arm failed during {name}")
            if not self.robot.has_grasp_contact():
                if name == "place approach":
                    # A spontaneous release over the destination is accepted
                    # only when the existing stable-placement verifier confirms it.
                    self.robot.command_gripper(0.08)
                    self.env.step(self.config.verification_steps)
                    if verify_stable(self.env, self.config):
                        self._set_phase(TaskPhase.COMPLETE)
                        return self._result(
                            started,
                            True,
                            "verified stable placement after release over target",
                        )
                event = next(
                    (
                        item
                        for item in reversed(self.monitor.events)
                        if item.event_type.value == "object_dropped"
                    ),
                    None,
                )
                recovered = self._recover_with_alternatives(event) if event is not None else False
                if not recovered:
                    self._set_phase(TaskPhase.FAILED)
                    return self._result(started, False, f"cube was lost during {name}")
                self._set_phase(TaskPhase.CARRYING)
                if not self._move_waypoint(point, grasp_rotation, maintain_grasp=True):
                    self._set_phase(TaskPhase.FAILED)
                    return self._result(started, False, f"arm failed after regrasp during {name}")
            # A controlled scene revision is informational, but the current goal
            # is replanned from the refreshed state before the next waypoint.
            scene_event = next(
                (
                    item
                    for item in reversed(self.monitor.events)
                    if item.event_type.value == "scene_changed"
                ),
                None,
            )
            if scene_event is not None and scene_event.step not in {
                item.event_step for item in self.recovery_attempts
            }:
                attempt = self.recovery.recover(scene_event, self)
                self.recovery_attempts.append(attempt)
        self.robot.command_gripper(0.08)
        self.env.step(250)
        verify_log = logging.getLogger("recovera.VERIFY")
        verify_log.info("checking final cube placement and stability")
        success = verify_stable(self.env, self.config)
        verify_log.info(
            "placement result=%s cube_position=(%.3f, %.3f, %.3f)",
            "verified" if success else "failed",
            *self.robot.cube_position,
        )
        self._set_phase(TaskPhase.COMPLETE if success else TaskPhase.FAILED)
        return self._result(
            started,
            success,
            "verified stable placement" if success else "cube did not remain in target",
        )

    def _move_waypoint(self, point, rotation, *, maintain_grasp=False):
        if self.scenario != "obstacle-replan" or maintain_grasp:
            return self.robot.move_to(
                point,
                self.config.max_steps_per_waypoint,
                self.config.position_tolerance,
                target_rotation=rotation,
                max_joint_step=0.025,
                maintain_grasp=maintain_grasp,
            )
        planner = PandaJointPlanner(self.env, self.robot, seed=self.config.random_seed, max_iterations=300)
        gate = TrajectoryValidator(planner)
        rng = np.random.default_rng(self.config.random_seed)
        goals = [solve_site_ik(self.robot, point, rotation)]
        goals.extend(
            solve_site_ik(
                self.robot,
                point,
                rotation,
                initial_joint_state=rng.uniform(planner.lower, planner.upper),
            )
            for _ in range(24)
        )
        plan = None
        last_reason = "IK found no waypoint configuration"
        for goal in goals:
            if goal is None:
                continue
            candidate = planner.plan(
                self.robot.joint_positions,
                goal,
                allow_initial_finger_contact=True,
            )
            if candidate.status is PlanStatus.SUCCESS:
                plan = candidate
                break
            last_reason = candidate.reason
        if plan is None:
            log.warning("obstacle replanning failed: %s", last_reason)
            return False
        validation = gate.validate(plan, self.robot.joint_positions)
        if not validation.valid:
            log.warning("obstacle trajectory rejected: %s", validation.reason)
            return False
        log.info("obstacle-aware route accepted: %s points=%d", plan.reason, len(plan.trajectory.points))
        done, reason = execute_trajectory(self.robot, plan.trajectory)
        if not done:
            log.warning("obstacle trajectory execution failed: %s", reason)
        return done

    def _recover_with_alternatives(self, event) -> bool:
        """Bounded alternative loop; only LLM-mode failures try another candidate."""
        excluded: set[str] = set()
        allow_same_event_retry = False
        while True:
            attempt = self.recovery.recover(
                event,
                self,
                allow_same_event_retry=allow_same_event_retry,
                exclude_strategy_ids=excluded,
            )
            self.recovery_attempts.append(attempt)
            if attempt.resumed_original_task:
                return True
            if attempt.strategy is not None and attempt.strategy.value == "safe_stop":
                return False
            if attempt.strategy is None:
                return False
            excluded.add(attempt.strategy.value)
            remaining = set(attempt.candidate_strategy_ids) - excluded
            retryable_failure = attempt.terminal_reason in {
                "planning_failed",
                "validation_failed",
                "execution_failed",
                "regrasp_verification_failed",
                "recovery_pose_not_reached",
            }
            if (
                not self.config.llm_enabled
                or not retryable_failure
                or not remaining
                or len(self.recovery_attempts) >= self.config.max_recovery_attempts
            ):
                return False
            allow_same_event_retry = True

    def _result(self, started: float, success: bool, diagnostic: str, *, status=None, classification=None) -> TrialResult:
        cube = self.robot.cube_position
        if success:
            outcome = "task_success"
        elif (
            self.recovery_attempts
            and self.recovery_attempts[-1].outcome_classification == "safe_termination"
        ):
            outcome = "safe_stop"
        elif self.recovery_attempts and not any(
            attempt.resumed_original_task for attempt in self.recovery_attempts
        ):
            outcome = "recovery_failure"
        else:
            outcome = "task_failure"
        return TrialResult(
            status=status or (TrialStatus.SUCCESS if success else TrialStatus.SAFE_STOP if outcome == "safe_stop" else TrialStatus.FAILURE),
            success=success,
            outcome_classification=classification or outcome,
            elapsed_seconds=time.monotonic() - started,
            execution_duration_seconds=time.monotonic() - started,
            recovery_seconds=sum(
                attempt.recovery_duration_seconds or 0.0
                for attempt in self.recovery_attempts
            ),
            recovery_attempt_count=len(self.recovery_attempts),
            replan_count=sum(
                attempt.strategy is not None
                and attempt.strategy.value == "refresh_and_replan"
                for attempt in self.recovery_attempts
            ),
            simulated_seconds=float(self.env.data.time),
            cube_final_position=tuple(float(x) for x in cube),
            target_position=self.config.target_position,
            trial_id=self.trial_id,
            scenario=self.scenario,
            random_seed=self.config.random_seed,
            initial_configuration={
                "cube_position": list(self.config.cube_initial_position),
                "target_position": list(self.config.target_position),
                "timestep": self.config.timestep,
                "scenario_id": self.scenario,
                "obstacle": (
                    {
                        "type": get_scenario(self.scenario).obstacle_asset or "box",
                        "position": list(get_scenario(self.scenario).obstacle_position),
                        "half_size": (
                            list(get_scenario(self.scenario).obstacle_half_size)
                            if get_scenario(self.scenario).obstacle_half_size is not None
                            else None
                        ),
                        "scale": get_scenario(self.scenario).obstacle_scale,
                    }
                    if get_scenario(self.scenario).obstacle_position is not None
                    else None
                ),
            },
            injected_failure=(
                "first_grasp_contact_disabled"
                if self.config.demo_fail_first_grasp
                else get_scenario(self.config.scenario_id).injected_failure
            ),
            safe_stop_achieved=(status is TrialStatus.SAFE_STOP or outcome == "safe_stop"),
            diagnostics=(diagnostic,),
            events=self.monitor.events,
            recovery_attempts=tuple(self.recovery_attempts),
        )
