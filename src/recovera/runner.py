"""Single-trial lifecycle and live-viewer integration."""

import logging
import time
from contextlib import ExitStack
from uuid import uuid4

from mujoco import FatalError

from recovera.config import RecoveraConfig
from recovera.environment import MujocoEnvironment
from recovera.results import TrialResult
from recovera.robot import PandaAdapter
from recovera.scenarios import get_scenario
from recovera.task import PickAndPlaceTask

log = logging.getLogger("recovera.SIMULATION")


def run_trial(config: RecoveraConfig) -> TrialResult:
    """Initialize, run, summarize, and close one headless or viewed trial."""
    trial_id = uuid4().hex[:10]
    mode = "viewer" if config.viewer else "headless"
    scenario = config.scenario_id if config.scenario_id != "baseline" else (
        "recovery-demo"
        if config.demo_fail_first_grasp
        else "drop-demo"
        if config.demo_drop_after_grasp
        else "baseline"
    )
    started = time.monotonic()
    log.info("trial=%s Recovera startup mode=%s scenario=%s", trial_id, mode, scenario)
    log.info(
        "trial=%s config model=%s timestep=%.4fs max_recovery_attempts=%d "
        "llm_enabled=%s llm_provider=%s llm_model=%s llm_fallback=%s",
        trial_id,
        config.model_path.expanduser().resolve(),
        config.timestep,
        config.max_recovery_attempts,
        config.llm_enabled,
        config.llm_provider if config.llm_enabled else "disabled",
        config.llm_model
        if config.llm_enabled and config.llm_provider == "ollama"
        else "fixed-mock"
        if config.llm_enabled
        else "n/a",
        config.llm_fallback,
    )
    env = MujocoEnvironment(config)
    log.info("trial=%s MuJoCo model and scene loaded", trial_id)
    summary_logged = False
    try:
        task = PickAndPlaceTask(env, PandaAdapter(env), config, trial_id=trial_id)
        log.info("trial=%s monitoring and recovery initialized", trial_id)
        if not config.viewer:
            result = task.run()
        else:
            from mujoco import viewer as mujoco_viewer

            log.info("trial=%s opening native MuJoCo passive viewer", trial_id)
            viewer_stack = ExitStack()
            try:
                viewer = viewer_stack.enter_context(
                    mujoco_viewer.launch_passive(env.model, env.data)
                )
            except (RuntimeError, FatalError) as exc:
                viewer_stack.close()
                raise RuntimeError(
                    "Could not launch the MuJoCo viewer. On macOS use the bundled mjpython executable."
                ) from exc
            try:
                env.viewer = viewer
                camera = get_scenario(config.scenario_id)
                if camera.viewer_lookat is not None:
                    # Frame the pickup, lamp, and destination together in the
                    # initial window instead of relying on MuJoCo's generic view.
                    viewer.cam.lookat[:] = camera.viewer_lookat
                    viewer.cam.distance = camera.viewer_distance
                    viewer.cam.azimuth = camera.viewer_azimuth
                    viewer.cam.elevation = camera.viewer_elevation
                viewer.sync()
                if not viewer.is_running():
                    raise RuntimeError("MuJoCo viewer window did not stay open")
                result = task.run()
                _log_trial_summary(trial_id, result, time.monotonic() - started)
                summary_logged = True
                log.info("trial=%s task loop ended; close the viewer window to exit", trial_id)
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.05)
                env.viewer = None
            finally:
                viewer_stack.close()
        if not summary_logged:
            _log_trial_summary(trial_id, result, time.monotonic() - started)
        return result
    finally:
        env.viewer = None
        env.close()


def _log_trial_summary(trial_id: str, result: TrialResult, wall_seconds: float) -> None:
    event_types = [event.event_type.value for event in result.events]
    planning_successes = sum(
        attempt.planning_status is not None and attempt.planning_status.value == "success"
        for attempt in result.recovery_attempts
    )
    planning_failures = sum(
        attempt.planning_status is not None and attempt.planning_status.value != "success"
        for attempt in result.recovery_attempts
    )
    llm_attempts = [a for a in result.recovery_attempts if a.llm_provider is not None]
    llm_accepted = sum(a.llm_proposal_accepted is True for a in llm_attempts)
    llm_latency = sum(a.llm_latency_seconds or 0.0 for a in llm_attempts)
    recovery_outcome = (
        "verified"
        if any(a.resumed_original_task for a in result.recovery_attempts)
        else "not attempted"
        if not result.recovery_attempts
        else "not verified"
    )
    summary_log = log.info if result.success else log.warning
    summary_log(
        "trial=%s summary task=%s task_outcome=%s recovery=%s attempts=%d events=%s "
        "planning_successes=%d planning_failures=%d llm_requests=%d "
        "llm_accepted=%d llm_latency=%.3fs sim_time=%.3fs wall_time=%.3fs reason=%s",
        trial_id,
        result.status.value,
        result.outcome_classification,
        recovery_outcome,
        len(result.recovery_attempts),
        event_types or "none",
        planning_successes,
        planning_failures,
        len(llm_attempts),
        llm_accepted,
        llm_latency,
        result.simulated_seconds,
        wall_seconds,
        "; ".join(result.diagnostics),
    )
