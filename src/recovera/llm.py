"""Bounded local-LLM strategy proposal contract and Ollama provider."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from recovera.monitoring import EventType, TaskPhase


class FailureSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    event_type: EventType
    step: int = Field(ge=0)
    sim_time: float = Field(ge=0)
    entity_id: str = Field(max_length=80)
    explanation: str = Field(max_length=240)


class RecoveryAttemptSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    event_type: EventType
    strategy_id: str | None = Field(default=None, max_length=80)
    verification: str = Field(max_length=80)
    terminal_reason: str | None = Field(default=None, max_length=120)


class StrategyCandidate(BaseModel):
    """Application-authored action candidate; descriptions never encode controls."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    strategy_id: str = Field(min_length=1, max_length=80)
    target_subgoal: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=180)
    prerequisites: tuple[str, ...] = Field(max_length=5)
    requires_replanning: bool


class RecoveryRequest(BaseModel):
    """Bounded observed state; no commands, trajectories, controls, or raw simulator arrays."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    request_id: str = Field(min_length=1, max_length=80)
    task_phase: TaskPhase
    original_task_goal_xyz: tuple[float, float, float]
    failure: FailureSummary
    last_task_action: str | None = Field(default=None, max_length=160)
    last_task_action_outcome: str | None = Field(default=None, max_length=160)
    observed_step: int = Field(ge=0)
    observed_sim_time: float = Field(ge=0)
    cube_position_xyz: tuple[float, float, float]
    cube_orientation_wxyz: tuple[float, float, float, float]
    bilateral_finger_contact: bool
    robot_joint_positions_rad: tuple[float, ...] = Field(min_length=7, max_length=7)
    scene_revision: int = Field(ge=0)
    last_scene_change: str | None = Field(default=None, max_length=120)
    state_source: str = "privileged MuJoCo observation"
    state_valid: bool
    prerequisites: dict[str, bool]
    candidates: tuple[StrategyCandidate, ...] = Field(min_length=1, max_length=4)
    candidate_feasibility: dict[str, str] = Field(default_factory=dict, max_length=4)
    previous_attempts: tuple[RecoveryAttemptSummary, ...] = Field(max_length=5)

    @property
    def permitted_strategy_ids(self) -> tuple[str, ...]:
        return tuple(candidate.strategy_id for candidate in self.candidates)


class StrategyProposal(BaseModel):
    """Untrusted high-level proposal. It cannot encode joint targets or executable actions."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    request_id: str = Field(min_length=1, max_length=80)
    strategy_id: str | None = Field(default=None, max_length=80)
    target_subgoal: str | None = Field(default=None, max_length=80)
    prerequisites: tuple[str, ...] = Field(
        max_length=5,
        description=(
            "Additional requirements observed by the model; candidate prerequisites are "
            "enforced independently by the application."
        ),
    )
    rationale: str = Field(min_length=1, max_length=400)
    requires_replanning: bool
    abstain: bool = False


class ProposalValidation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    accepted: bool
    reason: str


class StrategyProvider(Protocol):
    """Provider interface; implementations return data and have no robot/environment access."""

    provider_name: str
    model_name: str
    inference_mode: str

    def propose(
        self, request: RecoveryRequest, *, timeout_seconds: float | None = None
    ) -> StrategyProposal: ...


class LLMProviderError(RuntimeError):
    pass


class LLMProviderTimeout(LLMProviderError):
    pass


class OllamaProvider:
    """Offline-only HTTP client for an already-running local Ollama service."""

    provider_name = "ollama"
    inference_mode = "real"

    def __init__(
        self,
        *,
        endpoint: str = "http://127.0.0.1:11434",
        model: str = "qwen3:4b",
        timeout_seconds: float = 8.0,
        temperature: float = 0.0,
        num_predict: int = 192,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.num_predict = num_predict

    def check_model_available(self, timeout_seconds: float | None = None) -> tuple[bool, str]:
        """Read local tags only. This method never pulls or installs a model."""
        request = urllib.request.Request(f"{self.endpoint}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds if timeout_seconds is None else timeout_seconds,
            ) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return False, f"Ollama unavailable: {exc}"
        names = {str(item.get("name", "")) for item in payload.get("models", [])}
        if self.model_name not in names:
            return (
                False,
                f"model {self.model_name!r} is not present locally; no download was attempted",
            )
        return True, f"Ollama model {self.model_name!r} is available locally"

    def propose(
        self, request: RecoveryRequest, *, timeout_seconds: float | None = None
    ) -> StrategyProposal:
        prompt = (
            "Choose one permitted high-level recovery strategy for this robot task, or abstain "
            "when state is uncertain or no candidate is appropriate. "
            "The state is privileged MuJoCo observation, not camera perception. "
            "Do not invent coordinates, joint commands, trajectories, code, or tool calls. "
            "Return only an object matching the supplied JSON schema. "
            "Copy request_id exactly. Use a target_subgoal and prerequisites consistent "
            "with a listed candidate. Candidate prerequisites are trusted gates that the "
            "application checks independently; report only additional satisfied prerequisites. "
            "For abstention, set abstain=true, strategy_id and "
            "target_subgoal to null and prerequisites to an empty list. Set "
            "requires_replanning to the value allowed by the schema. Rationale is "
            "descriptive and never executable.\n" + request.model_dump_json()
        )
        schema = StrategyProposal.model_json_schema()
        schema["properties"]["strategy_id"] = {
            "anyOf": [
                {"type": "string", "enum": list(request.permitted_strategy_ids)},
                {"type": "null"},
            ]
        }
        schema["properties"]["target_subgoal"] = {
            "anyOf": [
                {"type": "string", "enum": [item.target_subgoal for item in request.candidates]},
                {"type": "null"},
            ]
        }
        schema["properties"]["requires_replanning"] = {
            "enum": sorted({item.requires_replanning for item in request.candidates})
        }
        schema["properties"]["prerequisites"] = {
            "type": "array",
            "items": {"type": "string", "enum": list(request.prerequisites)},
            "maxItems": 5,
            "uniqueItems": True,
        }
        body = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        encoded = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.endpoint}/api/generate",
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_data = json.loads(response.read())
            raw = response_data["response"]
            return StrategyProposal.model_validate_json(raw)
        except TimeoutError as exc:
            raise LLMProviderTimeout("Ollama strategy request timed out") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise LLMProviderTimeout("Ollama strategy request timed out") from exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                raise LLMProviderError(
                    f"Ollama endpoint or model {self.model_name!r} is unavailable; no download attempted"
                ) from exc
            raise LLMProviderError(f"Ollama request failed: {exc.reason}") from exc
        except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
            raise LLMProviderError(f"Ollama returned malformed structured output: {exc}") from exc


class MockStrategyProvider:
    """Explicit non-inference provider for CLI demonstrations and tests."""

    provider_name = "mock"
    model_name = "fixed-mock"
    inference_mode = "mock"

    def __init__(self, strategy_id: str | None = None):
        self.strategy_id = strategy_id

    def propose(
        self, request: RecoveryRequest, *, timeout_seconds: float | None = None
    ) -> StrategyProposal:
        del timeout_seconds
        candidate = next(
            (
                item
                for item in request.candidates
                if self.strategy_id is None or item.strategy_id == self.strategy_id
            ),
            None,
        )
        if candidate is None:
            return StrategyProposal(
                request_id=request.request_id,
                strategy_id=None,
                target_subgoal=None,
                prerequisites=(),
                rationale="No requested mock strategy is permitted in this state.",
                requires_replanning=False,
                abstain=True,
            )
        return StrategyProposal(
            request_id=request.request_id,
            strategy_id=candidate.strategy_id,
            target_subgoal=candidate.target_subgoal,
            prerequisites=candidate.prerequisites,
            rationale="Explicit mock provider selected this permitted candidate.",
            requires_replanning=candidate.requires_replanning,
        )


def new_request_id() -> str:
    return uuid4().hex


def validate_proposal(request: RecoveryRequest, proposal: StrategyProposal) -> ProposalValidation:
    if proposal.request_id != request.request_id:
        return ProposalValidation(
            accepted=False, reason="proposal request_id is stale or mismatched"
        )
    if proposal.abstain:
        if (
            proposal.strategy_id is not None
            or proposal.target_subgoal is not None
            or proposal.prerequisites
        ):
            return ProposalValidation(accepted=False, reason="abstention contains an action")
        return ProposalValidation(accepted=True, reason="model explicitly abstained")
    if proposal.strategy_id not in request.permitted_strategy_ids:
        return ProposalValidation(
            accepted=False, reason="strategy_id is not in the permitted allowlist"
        )
    candidate = next(
        item for item in request.candidates if item.strategy_id == proposal.strategy_id
    )
    if proposal.target_subgoal != candidate.target_subgoal:
        return ProposalValidation(accepted=False, reason="target_subgoal conflicts with strategy")
    if any(not request.prerequisites.get(name, False) for name in candidate.prerequisites):
        return ProposalValidation(accepted=False, reason="candidate prerequisite is no longer met")
    if any(not request.prerequisites.get(name, False) for name in proposal.prerequisites):
        return ProposalValidation(accepted=False, reason="proposal requires an unmet prerequisite")
    if not request.state_valid or not request.prerequisites.get("world_state_valid", False):
        return ProposalValidation(accepted=False, reason="request world state is invalid")
    if proposal.requires_replanning != candidate.requires_replanning:
        return ProposalValidation(
            accepted=False, reason="proposal replanning requirement conflicts with candidate"
        )
    return ProposalValidation(
        accepted=True, reason="proposal matches request, prerequisites, and allowlist"
    )
