"""Deterministic monitoring of privileged MuJoCo state during task execution."""

from enum import Enum

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict


class TaskPhase(str, Enum):
    INITIALIZING = "initializing"
    APPROACHING = "approaching"
    GRASPING = "grasping"
    CARRYING = "carrying"
    PLACING = "placing"
    COMPLETE = "complete"
    FAILED = "failed"


class TaskExpectation(BaseModel):
    model_config = ConfigDict(frozen=True)
    phase: TaskPhase
    active: bool = True
    grasp_attempted: bool = False
    grasp_confirmed: bool = False
    previously_held: bool = False


class WorldSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    step: int
    sim_time: float
    cube_position: tuple[float, float, float]
    cube_orientation: tuple[float, float, float, float]
    cube_linear_velocity: tuple[float, float, float]
    cube_angular_velocity: tuple[float, float, float]
    gripper_position: tuple[float, float, float]
    bilateral_finger_contact: bool
    scene_revision: int
    valid: bool = True
    diagnostic: str | None = None


class EventType(str, Enum):
    FAILED_GRASP = "failed_grasp"
    OBJECT_DROPPED = "object_dropped"
    SCENE_CHANGED = "scene_changed"


class MonitoringEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    event_type: EventType
    step: int
    sim_time: float
    entity_id: str
    expected_phase: TaskPhase
    explanation: str
    evidence: dict[str, object]


class WorldStateExtractor:
    """Translate named MuJoCo state into stable, validated values."""

    def __init__(self, env):
        self.env = env
        model = env.model
        self.cube_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geom = self._id(mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.fingers = {
            self._id(mujoco.mjtObj.mjOBJ_BODY, "left_finger"),
            self._id(mujoco.mjtObj.mjOBJ_BODY, "right_finger"),
        }
        self.grasp_site = self._id(mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        self.cube_dof = int(model.jnt_dofadr[self._id(mujoco.mjtObj.mjOBJ_JOINT, "cube_free")])

    def _id(self, kind, name: str) -> int:
        value = mujoco.mj_name2id(self.env.model, kind, name)
        if value < 0:
            raise ValueError(f"Required model {kind.name} {name!r} is missing")
        return int(value)

    def capture(self) -> WorldSnapshot:
        data = self.env.data
        position = data.xpos[self.cube_body].copy()
        orientation = data.xquat[self.cube_body].copy()
        linear, angular = (
            data.qvel[self.cube_dof : self.cube_dof + 3].copy(),
            data.qvel[self.cube_dof + 3 : self.cube_dof + 6].copy(),
        )
        gripper = data.site_xpos[self.grasp_site].copy()
        contacts: set[int] = set()
        for contact in data.contact[: data.ncon]:
            if self.cube_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.cube_geom else contact.geom1
            body = int(self.env.model.geom_bodyid[other])
            if body in self.fingers:
                contacts.add(body)
        arrays = (position, orientation, linear, angular, gripper)
        valid = all(np.isfinite(item).all() for item in arrays) and np.isfinite(data.time)
        return WorldSnapshot(
            step=round(data.time / self.env.model.opt.timestep),
            sim_time=float(data.time),
            cube_position=tuple(float(x) for x in position),
            cube_orientation=tuple(float(x) for x in orientation),
            cube_linear_velocity=tuple(float(x) for x in linear),
            cube_angular_velocity=tuple(float(x) for x in angular),
            gripper_position=tuple(float(x) for x in gripper),
            bilateral_finger_contact=contacts == self.fingers,
            scene_revision=self.env.scene_revision,
            valid=bool(valid),
            diagnostic=None if valid else "non-finite MuJoCo observation",
        )


class WorldStateMonitor:
    """Per-trial detector state. Invalid samples never advance a failure detector."""

    def __init__(self, env, grasp_timeout_steps: int = 1200, drop_confirmation_steps: int = 25):
        self.env = env
        self.extractor = WorldStateExtractor(env)
        self.grasp_timeout_steps = grasp_timeout_steps
        self.drop_confirmation_steps = drop_confirmation_steps
        self.reset()

    def reset(self) -> None:
        self._grasp_start: int | None = None
        self._held = False
        self._drop_start: int | None = None
        self._drop_reported = False
        self._scene_revision = self.env.scene_revision
        self._events: list[MonitoringEvent] = []

    def begin_grasp_attempt(self) -> None:
        """Start a new grasp timeout window without clearing this trial's event log."""
        self._grasp_start = None

    @property
    def events(self) -> tuple[MonitoringEvent, ...]:
        return tuple(self._events)

    def record_event(self, event: MonitoringEvent) -> None:
        """Append a controlled scenario event to the per-trial event log."""
        self._events.append(event)

    def observe(self, expectation: TaskExpectation) -> tuple[MonitoringEvent, ...]:
        snapshot = self.extractor.capture()
        if not snapshot.valid:
            return ()
        emitted: list[MonitoringEvent] = []
        if snapshot.scene_revision != self._scene_revision:
            change = getattr(self.env, "last_scene_change", None) or ("scene", "update")
            emitted.append(
                MonitoringEvent(
                    event_type=EventType.SCENE_CHANGED,
                    step=snapshot.step,
                    sim_time=snapshot.sim_time,
                    entity_id=change[0],
                    expected_phase=expectation.phase,
                    explanation="A controlled scene update was recorded.",
                    evidence={
                        "scene_revision": snapshot.scene_revision,
                        "operation": change[1],
                    },
                )
            )
            self._scene_revision = snapshot.scene_revision

        if (
            expectation.active
            and expectation.phase is TaskPhase.GRASPING
            and expectation.grasp_attempted
        ):
            if expectation.grasp_confirmed:
                self._held = True
            elif self._grasp_start is None:
                self._grasp_start = snapshot.step
            elif (
                not snapshot.bilateral_finger_contact
                and snapshot.step - self._grasp_start >= self.grasp_timeout_steps
            ):
                emitted.append(
                    MonitoringEvent(
                        event_type=EventType.FAILED_GRASP,
                        step=snapshot.step,
                        sim_time=snapshot.sim_time,
                        entity_id="cube",
                        expected_phase=expectation.phase,
                        explanation="Bilateral fingertip contact was not established before the grasp timeout.",
                        evidence={
                            "bilateral_finger_contact": False,
                            "elapsed_steps": snapshot.step - self._grasp_start,
                        },
                    )
                )
                self._grasp_start = snapshot.step

        if (
            (self._held or expectation.previously_held)
            and expectation.active
            and expectation.phase in (TaskPhase.CARRYING, TaskPhase.PLACING)
        ):
            relative = np.linalg.norm(
                np.asarray(snapshot.cube_position) - np.asarray(snapshot.gripper_position)
            )
            lost = not snapshot.bilateral_finger_contact and relative > 0.055
            if lost:
                if self._drop_start is None:
                    self._drop_start = snapshot.step
                elif (
                    not self._drop_reported
                    and snapshot.step - self._drop_start >= self.drop_confirmation_steps
                ):
                    emitted.append(
                        MonitoringEvent(
                            event_type=EventType.OBJECT_DROPPED,
                            step=snapshot.step,
                            sim_time=snapshot.sim_time,
                            entity_id="cube",
                            expected_phase=expectation.phase,
                            explanation="Previously held cube lost bilateral contact and separated from the gripper persistently.",
                            evidence={
                                "bilateral_finger_contact": False,
                                "gripper_separation_m": float(relative),
                                "confirmation_steps": snapshot.step - self._drop_start,
                            },
                        )
                    )
                    self._drop_reported = True
            else:
                self._drop_start = None
                if snapshot.bilateral_finger_contact:
                    self._drop_reported = False
        self._events.extend(emitted)
        return tuple(emitted)
