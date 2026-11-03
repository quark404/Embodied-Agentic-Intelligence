import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cluster_eval.image_inventory import inspect_image_runtime


class ImageInventoryTests(unittest.TestCase):
    def test_image_metadata_does_not_import_simulator_or_execute_installers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / 'python'
            executable.touch()
            source = root / 'RoboDojo'
            source.mkdir()
            (source / 'README.md').write_text('reference')
            results = [SimpleNamespace(returncode=0, stdout=json.dumps({'python': '3.11.14', 'packages': {}})),
                       SimpleNamespace(returncode=128, stdout='')]
            with patch('cluster_eval.probe.check_execution_context'), \
                    patch('cluster_eval.image_inventory.SIM_PYTHON', executable), \
                    patch('cluster_eval.image_inventory.SOURCE_ROOT', source), \
                    patch('cluster_eval.image_inventory.subprocess.run', side_effect=results) as run:
                report = inspect_image_runtime()
            command = run.call_args_list[0].args[0]
            self.assertEqual(command[:4], [str(executable), '-I', '-B', '-c'])
            self.assertNotIn('import isaacsim', command[-1])
            self.assertFalse(report['modified_image'])
            self.assertFalse(report['native_evaluation_ready'])
            self.assertIsNone(report['source_git_revision'])
            self.assertIn('README.md', report['source_files'])

    def test_no_inspection_without_allocated_execution_context(self):
        with patch('cluster_eval.probe.check_execution_context', side_effect=ValueError('not allocated')), \
                patch('cluster_eval.image_inventory.subprocess.run') as run, self.assertRaises(ValueError):
            inspect_image_runtime()
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
