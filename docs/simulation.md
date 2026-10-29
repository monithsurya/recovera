# Simulation and benchmark notes

## Trial flow

Every scenario uses one `MujocoEnvironment`, `PandaAdapter`, `PickAndPlaceTask`, monitor, recovery coordinator, and final placement verifier. The CLI changes scenario configuration; it does not start a second controller. A headless run and a viewer run therefore use the same model, task logic, collision geometry, and result schema.

At startup, the runner loads the Panda MJCF and builds the floor, cube, target, and any scenario obstacle. The monitor observes completed simulation steps without advancing MuJoCo a second time. A confirmed event captures a fresh snapshot before recovery is considered. A proposal is useful only when its prerequisites hold and planning, trajectory validation, execution, and physical verification all succeed.

The result record distinguishes task success from safe termination. A resting cube left untouched after a verified blocked route is a `safe_stop`, not a successful placement. Each JSON record contains the initial setup, detector events, candidate feasibility, selected strategy and inference mode, plan and execution results, final cube pose, and timing.

## Obstacle scenario

The table-lamp asset is composed of 15 OBJ meshes converted from the original GLB model's Y-up coordinates to the scene's Z-up convention. They are visual only. Collision checking uses one cylinder for the shade and three capsules for the tripod legs. The MuJoCo collision masks for these proxies are the same masks read by the planner.

Scenario 3 raises a scene-change event for the configured, static obstacle. The recovery coordinator refreshes the snapshot and checks candidate postures. IK solutions are tested against MuJoCo collision state; the joint-space RRT checks its sampled configurations and trajectory validator checks the returned path again before execution. Seeded IK alternatives are also considered for waypoints that need a collision-free arm posture. The route carries the cube above the lamp at the configured transfer height, then lowers it over the separate green target. Placement is reported only after the normal stability check.

Scenario 4 is intentionally infeasible for the supported top-down grasp approach. Its wall fixture is checked by the planner before the task begins; it is not a claim inferred from the fixture's appearance.

## Other scenarios

- **Baseline:** no injected fault; verifies ordinary pick-and-place behavior.
- **Mid-Air Drop:** opens the gripper during lift. The monitor must confirm loss of the previously established bilateral grasp before recovery runs.
- **Blocked Route:** confirms that the required approach collides and keeps the cube resting in place.
- **Drop Beyond Reach:** injects a deterministic horizontal force and release during lift. The bounded grasp preflight rejects configurations that fail IK or collision checks and does not retry indefinitely.

## Reproducibility

Mock inference uses a fixed policy and is intended for deterministic tests and demonstrations. The obstacle scenario also uses a fixed random seed for its alternate IK samples and planner. MuJoCo simulation time, random seed, initial cube and target poses, obstacle pose, provider mode, and recovery attempts are written to the trial record.

The project contains no camera perception: poses and contacts come directly from the MuJoCo model. Scene revisions represent changes made through the environment's supported scene-update API; arbitrary edits to model arrays are not monitored. Planning collision checks are discrete samples, not continuous swept-volume proofs. The lamp's visual mesh is more detailed than its proxy collision shapes. Recovery while holding a cube is limited by the current model, in which the cube is not a rigidly attached robot payload.

## Reading a result

Use `status` and `success` together. `status=success` means the task passed placement verification. `status=safe_stop` means Recovera stopped with the object left in the verified resting state; `success` remains false. `diagnostics` gives the terminal reason, while `events` and `recovery_attempts` describe the route to that outcome. `inference_mode` separates real model inference from mock selection and deterministic fallback.
