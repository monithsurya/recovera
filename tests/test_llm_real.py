"""Explicit, opt-in smoke test against a real Ollama model on loopback."""

import json
import os
import time

import pytest

from recovera.llm import (
    FailureSummary,
    OllamaProvider,
    RecoveryRequest,
    StrategyCandidate,
    validate_proposal,
)
from recovera.monitoring import EventType, TaskPhase


@pytest.mark.real_llm
def test_real_ollama_structured_strategy_smoke() -> None:
    if os.getenv("RECOVERA_RUN_REAL_LLM") != "1":
        pytest.skip("set RECOVERA_RUN_REAL_LLM=1 to run real local inference")
    model = os.getenv("RECOVERA_LLM_MODEL", "qwen3:4b")
    endpoint = os.getenv("RECOVERA_LLM_ENDPOINT", "http://127.0.0.1:11434")
    timeout = float(os.getenv("RECOVERA_LLM_TIMEOUT", "60"))
    provider = OllamaProvider(endpoint=endpoint, model=model, timeout_seconds=timeout)
    available, message = provider.check_model_available(timeout_seconds=min(timeout, 5))
    if not available:
        pytest.skip(f"real Ollama smoke skipped: {message}")

    request = RecoveryRequest(
        request_id="real-smoke-request",
        task_phase=TaskPhase.GRASPING,
        original_task_goal_xyz=(0.45, 0.25, 0.0),
        failure=FailureSummary(
            event_type=EventType.FAILED_GRASP,
            step=1200,
            sim_time=2.4,
            entity_id="cube",
            explanation="The grasp attempt ended without bilateral fingertip contact.",
        ),
        last_task_action="top_down_grasp_at_calibrated_plus_8mm_x_offset",
        last_task_action_outcome="confirmed_no_bilateral_fingertip_contact",
        observed_step=1201,
        observed_sim_time=2.402,
        cube_position_xyz=(0.45, 0.0, 0.018),
        cube_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        bilateral_finger_contact=False,
        robot_joint_positions_rad=(0.0, 0.0, 0.0, -1.57, 0.0, 1.57, -0.78),
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
                description="Retry from the calibrated lateral offset.",
                prerequisites=("grasp_attempted",),
                requires_replanning=True,
            ),
            StrategyCandidate(
                strategy_id="centered_reapproach_and_regrasp",
                target_subgoal="reacquire_cube_centered",
                description="Retry from a centered end-effector approach.",
                prerequisites=("grasp_attempted",),
                requires_replanning=True,
            ),
        ),
        previous_attempts=(),
    )
    started = time.perf_counter()
    proposal = provider.propose(request, timeout_seconds=timeout)
    elapsed = time.perf_counter() - started
    validation = validate_proposal(request, proposal)
    evidence = {
        "provider": provider.provider_name,
        "model": model,
        "inference_mode": provider.inference_mode,
        "elapsed_seconds": round(elapsed, 3),
        "strategy_id": proposal.strategy_id,
        "abstain": proposal.abstain,
        "candidate_ids": list(request.permitted_strategy_ids),
        "reported_prerequisites": list(proposal.prerequisites),
        "accepted": validation.accepted,
        "validation_reason": validation.reason,
    }
    print(json.dumps(evidence, sort_keys=True))
    assert validation.accepted, evidence
