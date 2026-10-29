"""Independent outcome checks from simulation state."""

from recovera.config import RecoveraConfig
from recovera.environment import MujocoEnvironment


def is_cube_in_target(env: MujocoEnvironment, config: RecoveraConfig) -> bool:
    """Check cube footprint containment and resting height above the floor."""
    cube = env.body_position("cube")
    tx, ty, floor_z = config.target_position
    hx, hy = config.target_half_size
    half = config.cube_half_size
    return bool(
        abs(cube[0] - tx) + half <= hx
        and abs(cube[1] - ty) + half <= hy
        and abs(cube[2] - (floor_z + half)) <= 0.018
    )


def verify_stable(env: MujocoEnvironment, config: RecoveraConfig) -> bool:
    """Require placement to remain valid throughout the verification interval."""
    for _ in range(config.verification_steps):
        if env.viewer_closed:
            return False
        if not is_cube_in_target(env, config):
            return False
        env.step()
        if env.viewer_closed:
            return False
    return is_cube_in_target(env, config)
