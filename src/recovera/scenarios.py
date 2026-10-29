"""Reproducible scenario catalogue shared by the CLI and simulation stack."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    label: str
    description: str
    obstacle_position: tuple[float, float, float] | None = None
    obstacle_half_size: tuple[float, float, float] | None = None
    obstacle_asset: str | None = None
    obstacle_scale: float = 1.0
    cube_initial_position: tuple[float, float, float] | None = None
    target_position: tuple[float, float, float] | None = None
    viewer_lookat: tuple[float, float, float] | None = None
    viewer_distance: float = 1.85
    viewer_azimuth: float = 135.0
    viewer_elevation: float = -19.0
    injected_failure: str | None = None
    llm_default: bool = False


SCENARIOS = {
    "baseline": ScenarioDefinition(
        "baseline", "Normal Pick and Place", "Baseline task with no injected failure."
    ),
    "drop-demo": ScenarioDefinition(
        "drop-demo",
        "Mid-Air Drop — LLM-Guided Recovery",
        "Release during lift; detect the drop, regrasp, and resume placement.",
        injected_failure="mid_lift_gripper_release",
        llm_default=True,
    ),
    "obstacle-replan": ScenarioDefinition(
        "obstacle-replan",
        "Obstacle Avoidance — LLM-Guided Replanning",
        "A CC0 tripod table lamp blocks the tabletop transfer route; seeded alternate IK postures are checked and replanned around its shade and tripod.",
        obstacle_position=(0.45, 0.03, 0.0),
        obstacle_asset="tripod-table-lamp",
        obstacle_scale=0.65,
        cube_initial_position=(0.45, -0.20, 0.018),
        target_position=(0.45, 0.32, 0.0),
        viewer_lookat=(0.29, 0.04, 0.36),
        viewer_distance=1.65,
        viewer_azimuth=315.0,
        llm_default=True,
    ),
    "blocked-route": ScenarioDefinition(
        "blocked-route",
        "Blocked Route — Safe Stop or Safe Placement",
        "A wall blocks the only supported top-down grasp approach; planner-confirmed safe stop leaves the cube at rest.",
        obstacle_position=(0.45, 0.0, 0.13),
        obstacle_half_size=(0.13, 0.13, 0.13),
        llm_default=True,
    ),
    "beyond-reach": ScenarioDefinition(
        "beyond-reach",
        "Drop Beyond Reach — Reachability-Aware Recovery",
        "A force pulse during a 60 mm lift sends the released cube outside the Panda IK workspace.",
        injected_failure="mid_lift_force_and_release",
        llm_default=True,
    ),
    # Preserve the Chunk 3 controlled failed-grasp CLI scenario.
    "recovery-demo": ScenarioDefinition(
        "recovery-demo",
        "Failed Grasp Recovery Demo",
        "Suppress first grasp contact and recover using the existing coordinator.",
        injected_failure="initial_grasp_contact_suppressed",
    ),
}


def get_scenario(scenario_id: str) -> ScenarioDefinition:
    try:
        return SCENARIOS[scenario_id]
    except KeyError as exc:
        raise ValueError(f"unknown Recovera scenario: {scenario_id}") from exc


MENU_SCENARIO_IDS = (
    "baseline",
    "drop-demo",
    "obstacle-replan",
    "blocked-route",
    "beyond-reach",
)
