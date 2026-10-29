"""Validated experiment configuration; values are not physical safety checks."""

from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from recovera.scenarios import get_scenario


class RecoveraConfig(BaseModel):
    """Settings for the deterministic Panda experiment."""

    model_config = ConfigDict(frozen=True)
    model_path: Path = Path("assets/franka_emika_panda/panda.xml")
    scenario_id: Literal["baseline", "recovery-demo", "drop-demo", "obstacle-replan", "blocked-route", "beyond-reach"] = "baseline"
    random_seed: int = 17
    drop_force_newtons: float = Field(default=1.2, ge=0, le=2)
    timestep: float = Field(default=0.002, gt=0)
    max_steps_per_waypoint: int = Field(default=6000, gt=0)
    position_tolerance: float = Field(default=0.005, gt=0)
    cube_half_size: float = Field(default=0.018, gt=0)
    cube_initial_position: tuple[float, float, float] = (0.45, 0.0, 0.018)
    target_position: tuple[float, float, float] = (0.45, 0.25, 0.0)
    target_half_size: tuple[float, float] = (0.08, 0.08)
    verification_steps: int = Field(default=150, gt=0)
    max_recovery_attempts: int = Field(default=2, ge=0)
    demo_fail_first_grasp: bool = False
    demo_drop_after_grasp: bool = False
    viewer: bool = False
    llm_enabled: bool = False
    llm_provider: Literal["ollama", "mock"] = "ollama"
    llm_mock_strategy: str | None = Field(default=None, max_length=80)
    llm_model: str = Field(default="qwen3:4b", min_length=1, max_length=120)
    llm_endpoint: str = "http://127.0.0.1:11434"
    llm_timeout_seconds: float = Field(default=8.0, gt=0, le=60)
    llm_temperature: float = Field(default=0.0, ge=0, le=2)
    llm_num_predict: int = Field(default=192, ge=32, le=512)
    llm_fallback: Literal["deterministic", "stop"] = "deterministic"
    llm_total_budget_seconds: float = Field(default=16.0, gt=0, le=120)

    @model_validator(mode="before")
    @classmethod
    def apply_scenario_positions(cls, value):
        """Use scenario-specific poses unless the caller supplied its own."""
        if not isinstance(value, dict):
            return value
        values = dict(value)
        scenario = get_scenario(values.get("scenario_id", "baseline"))
        if scenario.cube_initial_position is not None:
            values.setdefault("cube_initial_position", scenario.cube_initial_position)
        if scenario.target_position is not None:
            values.setdefault("target_position", scenario.target_position)
        return values

    @field_validator("cube_initial_position", "target_position")
    @classmethod
    def finite_xyz(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if len(value) != 3 or any(not -2 <= x <= 2 for x in value):
            raise ValueError("positions must contain three coordinates within [-2, 2] metres")
        return value

    @field_validator("target_half_size")
    @classmethod
    def valid_target_size(cls, value: tuple[float, float]) -> tuple[float, float]:
        if len(value) != 2 or any(x <= 0 for x in value):
            raise ValueError("target_half_size must contain two positive values")
        return value

    @field_validator("llm_endpoint")
    @classmethod
    def local_llm_endpoint_only(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("LLM endpoint must use local HTTP loopback only")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("LLM endpoint must not contain credentials, query, or fragment")
        return value.rstrip("/")
