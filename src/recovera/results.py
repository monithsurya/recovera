"""Validated experiment outcomes."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from recovera.monitoring import MonitoringEvent
from recovera.recovery import RecoveryAttempt


class TrialStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    SAFE_STOP = "safe_stop"
    INCOMPLETE = "incomplete"


class TrialResult(BaseModel):
    """Structured task result, including simulation metrics and diagnostics."""

    model_config = ConfigDict(frozen=True)
    status: TrialStatus
    success: bool
    outcome_classification: str = "task_failure"
    elapsed_seconds: float
    execution_duration_seconds: float = 0.0
    recovery_seconds: float = 0.0
    recovery_attempt_count: int = 0
    replan_count: int = 0
    simulated_seconds: float
    cube_final_position: tuple[float, float, float]
    target_position: tuple[float, float, float]
    trial_id: str | None = None
    scenario: str | None = None
    random_seed: int = 17
    initial_configuration: dict[str, object] = Field(default_factory=dict)
    injected_failure: str | None = None
    safe_stop_achieved: bool = False
    diagnostics: tuple[str, ...] = ()
    events: tuple[MonitoringEvent, ...] = ()
    recovery_attempts: tuple[RecoveryAttempt, ...] = ()
