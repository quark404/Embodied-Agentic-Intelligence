import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from cluster_eval.baked_job import inspect_assets, main, prepare_runtime
from cluster_eval.isaac_render import CameraBufferNotReady, camera_rgb, capture_diagnostics, capture_ready_frames, kit_arguments
from cluster_eval.prepare import ROOT, validate_template


class BakedRuntimeTests(unittest.TestCase):
    def test_explicit_capture_waits_for_real_buffers_with_a_fixed_budget(self):
        content = np.zeros((96, 96, 4), dtype=np.uint8)
        camera = SimpleNamespace(get_rgba=Mock(side_effect=[np.array([]), content]))
        step = Mock()
        frames = capture_ready_frames({'front': camera}, step)
        self.assertEqual(step.call_count, 2)
        step.assert_called_with(rt_subframes=4, pause_timeline=False, delta_time=0.0)
        np.testing.assert_array_equal(frames['front'], content[:, :, :3])
        camera.get_rgba = Mock(return_value=np.array([]))
        step.reset_mock()
        with self.assertRaises(CameraBufferNotReady):
            capture_ready_frames({'front': camera}, step)
        self.assertEqual(step.call_count, 5)

    def test_camera_explicitly_requests_cpu_data_without_synthesizing_pixels(self):
        original = np.arange(96 * 96 * 4, dtype=np.uint8).reshape(96, 96, 4)
        camera = SimpleNamespace(get_rgba=Mock(return_value=original))
        result = camera_rgb(camera)
        camera.get_rgba.assert_called_once_with(device='cpu')
        np.testing.assert_array_equal(result, original[:, :, :3])
        camera.get_rgba.return_value = np.empty((0,), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, '"shape": \\[0\\]'):
            camera_rgb(camera)

    def test_render_asset_check_does_not_repeat_the_entire_dataset_walk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'Object').mkdir()
            (root / 'Robots').mkdir()
            with patch('cluster_eval.baked_job.os.walk') as walk:
                result = inspect_assets(root, full_scan=False)
            walk.assert_not_called()
            self.assertTrue(result['assets_exists'])
            self.assertFalse(result['content_identity_verified'])
            self.assertIsNone(result['files'])

    def test_capture_error_is_saved_before_kit_can_exit_during_shutdown(self):
        def terminate():
            raise SystemExit(0)

        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            with self.assertRaises(SystemExit), capture_diagnostics(SimpleNamespace(close=terminate), results):
                raise ValueError('camera frame failed')
            report = json.loads((results / 'camera-failure.json').read_text())
            self.assertEqual(report, {'error_type': 'ValueError', 'message': 'camera frame failed'})

    def test_kit_portable_root_is_explicit_and_never_uses_image_package_cache(self):
        runtime = Path('/job_data/eai_test/runtime')
        self.assertEqual(kit_arguments(runtime, executable='render', root_user=False),
                         ['render', '--portable', '--portable-root', str(runtime / 'kit')])
        self.assertEqual(kit_arguments(runtime, executable='render', root_user=True)[-1], '--allow-root')

    def test_baked_render_has_one_gpu_read_only_inputs_and_no_install_profile(self):
        config = json.loads((ROOT / 'configs/aidi_5090_baked_render.example.json').read_text())
        validate_template(config, profile='baked_render')
        for key, value in (('output_bucket', 'robot_lab2'), ('gpu_per_worker', 3), ('wall_time', 31)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_template({**config, key: value}, profile='baked_render')

    def test_prebuilt_entry_refuses_control_host_without_a_renderer(self):
        with patch('cluster_eval.baked_job.probe.main', side_effect=ValueError('not allocated')), \
                patch('cluster_eval.baked_job.render_process') as renderer, self.assertRaises(ValueError):
            main()
        renderer.assert_not_called()

    def test_runtime_requires_real_image_versions_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / 'runtime'
            with patch('cluster_eval.baked_job.validate_local_destination'), self.assertRaises(ValueError):
                prepare_runtime(runtime, {'inventory_passed': True})
            self.assertFalse(runtime.exists())

    def test_asset_metadata_does_not_claim_content_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / 'Assets'
            assets.mkdir()
            (assets / 'test.usd').write_bytes(b'test')
            report = inspect_assets(root)
            self.assertEqual(report['files'], 1)
            self.assertEqual(report['bytes'], 4)
            self.assertFalse(report['content_identity_verified'])
            self.assertFalse(report['count_and_size_match'])
            (assets / 'outside').symlink_to('/tmp')
            with self.assertRaises(ValueError):
                inspect_assets(root)


if __name__ == '__main__':
    unittest.main()
