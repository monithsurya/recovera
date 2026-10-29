"""Render a real transport frame from the obstacle-replanning scenario."""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import mujoco

from recovera.config import RecoveraConfig
from recovera.environment import MujocoEnvironment
from recovera.monitoring import TaskPhase
from recovera.robot import PandaAdapter
from recovera.scenarios import get_scenario
from recovera.task import PickAndPlaceTask


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/images/obstacle-replan-transfer.png"),
        help="PNG path for the captured transport frame",
    )
    args = parser.parse_args()
    config = RecoveraConfig(
        scenario_id="obstacle-replan",
        llm_enabled=True,
        llm_provider="mock",
    )
    env = MujocoEnvironment(config)
    try:
        robot = PandaAdapter(env)
        task = PickAndPlaceTask(env, robot, config)
        captured: dict[str, object] = {}

        def capture_transport_state() -> None:
            if captured or task.phase is not TaskPhase.CARRYING:
                return
            cube = robot.cube_position
            if -0.02 <= cube[1] <= 0.12 and cube[2] >= 0.35 and robot.has_grasp_contact():
                captured.update(
                    qpos=env.data.qpos.copy(),
                    qvel=env.data.qvel.copy(),
                    act=env.data.act.copy(),
                    ctrl=env.data.ctrl.copy(),
                    sim_time=float(env.data.time),
                    cube_position=cube.copy(),
                )

        env.add_step_observer(capture_transport_state)
        result = task.run()
        if not result.success:
            raise RuntimeError(f"Obstacle-replan example did not complete: {result.diagnostics}")
        if not captured:
            raise RuntimeError("No frame captured while carrying the cube past the lamp")

        frame = mujoco.MjData(env.model)
        frame.qpos[:] = captured["qpos"]
        frame.qvel[:] = captured["qvel"]
        frame.act[:] = captured["act"]
        frame.ctrl[:] = captured["ctrl"]
        frame.time = captured["sim_time"]
        mujoco.mj_forward(env.model, frame)

        env.model.vis.global_.offwidth = 1280
        env.model.vis.global_.offheight = 960
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera_config = get_scenario(config.scenario_id)
        camera.lookat[:] = camera_config.viewer_lookat
        camera.distance = camera_config.viewer_distance
        camera.azimuth = camera_config.viewer_azimuth
        camera.elevation = camera_config.viewer_elevation
        with mujoco.Renderer(env.model, height=960, width=1280) as renderer:
            renderer.update_scene(frame, camera=camera)
            pixels = renderer.render()

        height, width, _ = pixels.shape
        scanlines = b"".join(b"\0" + row.tobytes() for row in pixels)
        png = (
            b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0))
            + _png_chunk(b"IDAT", zlib.compress(scanlines, 9))
            + _png_chunk(b"IEND", b"")
        )
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(png)
        print(
            f"Saved {output} at t={captured['sim_time']:.2f}s; "
            f"cube position={captured['cube_position']}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
