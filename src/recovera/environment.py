"""MuJoCo initialization and simulation lifecycle."""

import hashlib
import time
from pathlib import Path

import mujoco
import numpy as np

from recovera.config import RecoveraConfig
from recovera.scenarios import get_scenario


class MujocoEnvironment:
    """Own a MuJoCo model/data pair with Panda, floor, cube, and target."""

    def __init__(self, config: RecoveraConfig):
        self.config = config
        model_path = config.model_path.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Panda model not found at {model_path}; configure a compatible MJCF."
            )
        source = model_path.read_text()
        hand = source.index('<body name="hand"')
        hand_end = source.index(">", hand) + 1
        source = (
            source[:hand_end]
            + '<site name="grasp_site" pos="0 0 0.103" size="0.001" rgba="0 0 0 0" group="4"/>'
            + source[hand_end:]
        )
        # Keep fingertip contact geometry active but remove the palm shell from
        # this deliberately top-down grasp, where it would block descent.
        source = source.replace(
            '<geom mesh="hand_c" class="collision"/>',
            '<geom mesh="hand_c" class="collision" contype="0" conaffinity="0"/>',
            1,
        )
        # The upstream finger collision meshes extend below the pad contact
        # frames in this top-down scene and hit the support plane before the
        # pads reach the block. Keep the five modeled fingertip pad colliders
        # on each finger as the contact surfaces for this task.
        source = source.replace(
            '<geom mesh="finger_0" class="collision"/>',
            '<geom mesh="finger_0" class="collision" contype="0" conaffinity="0"/>',
        )
        for pad in range(1, 6):
            source = source.replace(
                f'<default class="fingertip_pad_collision_{pad}">\n          <geom type="box"',
                f'<default class="fingertip_pad_collision_{pad}">\n          <geom type="box" friction="4 0.1 0.01"',
                1,
            )
        # Preserve the upstream position mapping while increasing grip
        # stiffness and damping for the lightweight tabletop cube.
        source = source.replace(
            'gainprm="0.01568627451 0 0" biasprm="0 -100 -10"',
            'gainprm="0.03137254902 0 0" biasprm="0 -200 -100"',
            1,
        )
        tx, ty, tz = config.target_position
        hx, hy = config.target_half_size
        side = config.cube_half_size
        obstacle = ""
        scenario = get_scenario(config.scenario_id)
        if scenario.obstacle_asset == "tripod-table-lamp":
            lamp_dir = model_path.parent.parent / "obstacles" / "table_lamp"
            lamp_colors = (
                "0.72 0.48 0.23 1",
                "0.55 0.34 0.18 1",
                "0.55 0.34 0.18 1",
                "0.55 0.34 0.18 1",
                "0.55 0.34 0.18 1",
                "0.55 0.34 0.18 1",
                "0.55 0.34 0.18 1",
                "0.84 0.79 0.67 1",
                "0.72 0.48 0.23 1",
                "0.72 0.48 0.23 1",
                "0.72 0.48 0.23 1",
                "0.72 0.48 0.23 1",
                "0.72 0.48 0.23 1",
                "0.72 0.48 0.23 1",
                "1 0.87 0.56 1",
            )
            mesh_assets = "\n".join(
                f'<mesh name="lamp_part_{index:02}" file="{lamp_dir / f"lamp_part_{index:02}.obj"}" scale="{scenario.obstacle_scale} {scenario.obstacle_scale} {scenario.obstacle_scale}"/>'
                for index in range(len(lamp_colors))
            )
            source = source.replace("  </asset>", f"{mesh_assets}\n  </asset>", 1)
            visual_meshes = "\n".join(
                f'<geom name="lamp_visual_{index:02}" type="mesh" mesh="lamp_part_{index:02}" contype="0" conaffinity="0" rgba="{color}"/>'
                for index, color in enumerate(lamp_colors)
            )
            position = " ".join(map(str, scenario.obstacle_position or (0, 0, 0)))
            scale = scenario.obstacle_scale
            collision_shapes = f"""
        <geom name="route_obstacle" type="cylinder" pos="0 0 {0.409 * scale}" size="{0.188 * scale} {0.097 * scale}" contype="1" conaffinity="1" rgba="0 0 0 0" group="4"/>
        <geom name="route_obstacle_leg_a" type="capsule" fromto="{-0.084 * scale} {-0.137 * scale} 0 {-0.005 * scale} {-0.021 * scale} {0.317 * scale}" size="{0.012 * scale}" contype="1" conaffinity="1" rgba="0 0 0 0" group="4"/>
        <geom name="route_obstacle_leg_b" type="capsule" fromto="{-0.084 * scale} {0.137 * scale} 0 {-0.005 * scale} {0.021 * scale} {0.317 * scale}" size="{0.012 * scale}" contype="1" conaffinity="1" rgba="0 0 0 0" group="4"/>
        <geom name="route_obstacle_leg_c" type="capsule" fromto="{0.155 * scale} 0 0 {0.026 * scale} 0 {0.317 * scale}" size="{0.012 * scale}" contype="1" conaffinity="1" rgba="0 0 0 0" group="4"/>
        """
            obstacle = f'<body name="table_lamp" pos="{position}">{visual_meshes}{collision_shapes}</body>'
        elif scenario.obstacle_position is not None and scenario.obstacle_half_size is not None:
            position = " ".join(map(str, scenario.obstacle_position))
            half_size = " ".join(map(str, scenario.obstacle_half_size))
            rgba = "0.8 0.15 0.15 1"
            obstacle = (
                f'<geom name="route_obstacle" type="box" pos="{position}" '
                f'size="{half_size}" contype="1" conaffinity="1" rgba="{rgba}"/>'
            )
        scene = f"""
    <geom name="recovera_floor" type="plane" size="2 2 0.1" contype="2" conaffinity="4" solref="0.005 1" solimp="0.99 0.999 0.001" rgba="0.75 0.77 0.8 1"/>
    <body name="cube" pos="{config.cube_initial_position[0]} {config.cube_initial_position[1]} {config.cube_initial_position[2]}"><freejoint name="cube_free"/><geom name="cube_geom" type="box" size="{side} {side} {side}" mass="0.02" contype="4" conaffinity="3" solref="0.005 1" solimp="0.99 0.999 0.001" rgba="0.95 0.32 0.12 1" friction="4 0.1 0.01"/></body>
    <body name="target" pos="{tx} {ty} {tz + 0.001}"><geom name="target_geom" type="box" size="{hx} {hy} 0.001" contype="0" conaffinity="0" rgba="0.15 0.75 0.3 0.35"/></body>
    {obstacle}
"""
        source = source.replace("</worldbody>", scene + "</worldbody>", 1)
        source = source.replace('meshdir="assets"', f'meshdir="{model_path.parent / "assets"}"', 1)
        self.scene_path = model_path.with_name("recovera_scene.xml")
        self.scene_path.write_text(source)
        cache_key = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        # Keep compiled models out of the project directory, which may be
        # cloud-synced and slow to hydrate on first access.
        self.cache_path = (
            Path.home() / ".cache" / "recovera" / "models" / f"recovera_scene_{cache_key}.mjb"
        )
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            if self.cache_path.is_file():
                self.model = mujoco.MjModel.from_binary_path(str(self.cache_path))
            else:
                self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
                mujoco.mj_saveModel(self.model, str(self.cache_path))
            self.model.opt.timestep = config.timestep
            self.data = mujoco.MjData(self.model)
        except (ValueError, mujoco.FatalError) as exc:
            raise RuntimeError(f"Could not initialize Panda MuJoCo scene: {exc}") from exc
        cube_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        if cube_joint < 0:
            raise RuntimeError("Generated scene is missing its cube free joint")
        self.cube_qpos_address = int(self.model.jnt_qposadr[cube_joint])
        self.cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "recovera_floor"
        )
        self.home_keyframe = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if self.home_keyframe < 0:
            raise RuntimeError("Panda model is missing its required named 'home' keyframe")
        self.viewer = None
        self.viewer_closed = False
        self.scene_revision = 0
        self.last_scene_change: tuple[str, str] | None = None
        self._step_observers = []
        self.reset()

    def reset(self) -> None:
        """Restore the model keyframe and configured cube pose."""
        self.viewer_closed = False
        if self.viewer is not None:
            with self.viewer.lock():
                self._reset_state()
        else:
            self._reset_state()

    def _reset_state(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_keyframe)
        self.scene_revision = 0
        self.last_scene_change = None
        self.data.qpos[self.cube_qpos_address : self.cube_qpos_address + 3] = (
            self.config.cube_initial_position
        )
        self.model.geom_contype[self.cube_geom_id] = 4
        self.model.geom_conaffinity[self.cube_geom_id] = 2
        mujoco.mj_forward(self.model, self.data)

    def set_robot_cube_contacts(self, enabled: bool) -> None:
        """Gate robot-cube contact while retaining cube-floor support and no robot-floor contact."""

        # Robot collision bit: 1; floor bit: 2; cube bit: 4. The cube accepts
        # robot contacts only during grasp, and the floor only contacts cubes.
        def update():
            self.model.geom_contype[self.cube_geom_id] = 4
            self.model.geom_conaffinity[self.cube_geom_id] = 3 if enabled else 2

        if self.viewer is not None:
            with self.viewer.lock():
                update()
        else:
            update()

    def add_step_observer(self, observer) -> None:
        """Observe each completed MuJoCo step without advancing simulation again."""
        if observer not in self._step_observers:
            self._step_observers.append(observer)

    def remove_step_observer(self, observer) -> None:
        if observer in self._step_observers:
            self._step_observers.remove(observer)

    def record_scene_change(self, entity_id: str, operation: str) -> int:
        """Record an explicit supported scene update (add/move/remove/activate)."""
        if not entity_id.strip() or operation not in {
            "add",
            "move",
            "remove",
            "activate",
            "deactivate",
        }:
            raise ValueError("scene change requires an entity id and a supported operation")
        self.scene_revision += 1
        self.last_scene_change = (entity_id, operation)
        return self.scene_revision

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            if self.viewer is not None and not self.viewer.is_running():
                self.viewer_closed = True
                self.viewer = None
                break
            step_started = time.perf_counter()
            if self.viewer is not None:
                with self.viewer.lock():
                    mujoco.mj_step(self.model, self.data)
                    for observer in tuple(self._step_observers):
                        observer()
            else:
                mujoco.mj_step(self.model, self.data)
                for observer in tuple(self._step_observers):
                    observer()
            if self.viewer is not None:
                self.viewer.sync()
                if not self.viewer.is_running():
                    self.viewer_closed = True
                    self.viewer = None
                    break
                remaining = self.config.timestep - (time.perf_counter() - step_started)
                if remaining > 0:
                    time.sleep(remaining)

    def body_position(self, name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise KeyError(f"Body {name!r} is not present in the model")
        return self.data.xpos[body_id].copy()

    def close(self) -> None:
        """Headless MuJoCo resources are released with their Python objects."""
