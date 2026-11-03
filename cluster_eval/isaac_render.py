"""Render a procedural scene on the allocated GPU, never a benchmark result."""

from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import sys

from cluster_eval.rendering import CAMERAS, RESOLUTION, validate_render_report
from cluster_eval.storage import local_result_path, persist_export, validate_local_destination


ROOT = Path(__file__).resolve().parents[1]


class CameraBufferNotReady(ValueError):
    pass


def kit_arguments(runtime: Path, *, executable: str, root_user: bool) -> list[str]:
    arguments = [executable, '--portable', '--portable-root', str(runtime / 'kit')]
    if root_user:
        arguments.append('--allow-root')
    return arguments


def camera_rgb(camera):
    import numpy as np

    rgba = camera.get_rgba(device='cpu')
    metadata = {'type': type(rgba).__name__, 'shape': list(getattr(rgba, 'shape', ())),
                'dtype': str(getattr(rgba, 'dtype', None))}
    print('EAI_CAMERA_BUFFER_V1 ' + json.dumps(metadata), flush=True)
    if rgba is None or isinstance(rgba, np.ndarray) and rgba.size == 0:
        raise CameraBufferNotReady('Empty camera buffer: ' + json.dumps(metadata))
    if not isinstance(rgba, np.ndarray) or rgba.shape != (RESOLUTION, RESOLUTION, 4) or rgba.dtype != np.uint8:
        raise ValueError('Invalid CPU RGBA8 camera buffer: ' + json.dumps(metadata))
    return np.ascontiguousarray(rgba[:, :, :3])


def capture_ready_frames(cameras: dict, capture_step) -> dict:
    for attempt in range(5):
        capture_step(rt_subframes=4, pause_timeline=False, delta_time=0.0)
        try:
            return {name: camera_rgb(camera) for name, camera in cameras.items()}
        except CameraBufferNotReady:
            if attempt == 4:
                raise
    raise RuntimeError('No capture attempt completed')


@contextmanager
def capture_diagnostics(application, results: Path):
    try:
        yield
    except Exception as error:
        failure = {'error_type': type(error).__name__, 'message': str(error)[:512]}
        persist_export(results / 'camera-failure.json', failure)
        print('EAI_CAMERA_FAILURE_V1 ' + json.dumps(failure), flush=True)
        raise
    finally:
        application.close()


def main() -> None:
    results = local_result_path(os.environ.get('EAI_RUN_ID', ''))
    runtime = validate_local_destination(results.parent / 'runtime', ROOT)
    (runtime / 'kit').mkdir(mode=0o700)
    sys.argv = kit_arguments(runtime, executable=sys.argv[0], root_user=os.getuid() == 0)
    from isaacsim import SimulationApp

    application = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting',
                                  'active_gpu': 0, 'physics_gpu': 0, 'multi_gpu': False,
                                  'width': RESOLUTION, 'height': RESOLUTION, 'fast_shutdown': True})
    with capture_diagnostics(application, results):
        import numpy as np
        from PIL import Image
        from pxr import UsdLux
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
        from isaacsim.sensors.camera import Camera
        import omni.replicator.core as replicator
        import carb.settings

        world = World(stage_units_in_meters=1.0)
        for name, position, scale, color in (
                ('floor', (0, 0, -0.05), (4, 4, 0.1), (0.4, 0.4, 0.4)),
                ('red', (-0.35, 0, 0.3), (0.4, 0.4, 0.6), (1, 0, 0)),
                ('blue', (0.35, 0.2, 0.2), (0.3, 0.3, 0.4), (0, 0, 1))):
            world.scene.add(FixedCuboid(prim_path=f'/World/{name}', name=name,
                position=np.array(position, dtype=float), scale=np.array(scale, dtype=float),
                color=np.array(color, dtype=float), size=1.0))
        light = UsdLux.DomeLight.Define(world.stage, '/World/Light')
        light.CreateIntensityAttr(1500)
        orientation = euler_angles_to_quats(np.array([0, 90, 0]), degrees=True)
        cameras = {name: Camera(prim_path=f'/World/{name}_camera', frequency=20,
                    position=np.array([offset, 0, 3], dtype=float), orientation=orientation,
                    resolution=(RESOLUTION, RESOLUTION))
                   for name, offset in zip(CAMERAS, (0, -0.3, 0.3))}
        world.reset()
        for camera in cameras.values():
            camera.initialize()
        for step in range(120):
            if not application.is_running():
                raise RuntimeError('Isaac application stopped before camera capture')
            world.step(render=True)
        print('EAI_RENDER_STATE_V1 ' + json.dumps({'capture_on_play':
              carb.settings.get_settings().get('/omni/replicator/captureOnPlay')}), flush=True)
        ready = capture_ready_frames(cameras, replicator.orchestrator.step)
        frames = {}
        for name, rgb in ready.items():
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format='PNG')
            content = buffer.getvalue()
            frames[name] = {'width': RESOLUTION, 'height': RESOLUTION, 'channels': 3,
                            'stddev': float(rgb.std()), 'mean': float(rgb.mean()),
                            'sha256': hashlib.sha256(content).hexdigest(),
                            'png_base64': base64.b64encode(content).decode()}
        report = {'run_id': os.environ['EAI_RUN_ID'], 'storage_mode': 'job_local',
                  'storage_origin_verified': True, 'render_passed': True,
                  'scene': 'procedural_rgb_cubes', 'frames': frames,
                  'kit_portable_root': str(runtime / 'kit'),
                  'native_evaluation_ready': False}
        validate_render_report(report)
        persist_export(results / 'camera-capture.json', report)


if __name__ == '__main__':
    main()
