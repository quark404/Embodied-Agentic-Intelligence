import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from cluster_eval.resource_probe import SAMPLE_LIMIT, WheelLinks, file_sample, index_sample, resource_inventory


class Response(io.BytesIO):
    status = 200

    def geturl(self):
        return 'https://pypi.hobot.cc/simple/uv/'


class ResourceProbeTests(unittest.TestCase):
    def test_uploaded_script_imports_without_repository_on_pythonpath(self):
        for name in ('probe.py', 'image_probe.py'):
            script = Path(__file__).resolve().parents[1] / 'cluster_eval' / name
            result = subprocess.run([sys.executable, '-I', '-B', str(script)], env={},
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('EAI_CLUSTER_CONFIRMED=1 is required', result.stderr)
            self.assertNotIn('ModuleNotFoundError', result.stderr)

    def test_index_resolves_relative_and_encoded_urls_without_accepting_yanked_or_credentials(self):
        parser = WheelLinks('https://pypi.hobot.cc/simple/torch/', 'torch-2.7.0+cu128-cp311-cp311-*x86_64.whl')
        filename = 'torch-2.7.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl'
        parser.feed(f'<a href="../../packages/{filename}#sha256=abc">torch</a>'
                    f'<a data-yanked="" href="https://pypi.hobot.cc/yanked/{filename}">bad</a>'
                    f'<a href="https://user:secret@pypi.hobot.cc/{filename}">bad</a>')
        self.assertEqual(parser.links, ['https://pypi.hobot.cc/packages/' + filename])

    def test_successful_index_requires_a_real_wheel_response(self):
        content = b'<a href="../../packages/uv-0.9.3-py3-none-manylinux_2_17_x86_64.whl">uv</a>'
        with patch('cluster_eval.resource_probe.direct_open', side_effect=[Response(content), Response(b'login page')]):
            result = index_sample('https://pypi.hobot.cc/simple/uv/', 'uv-0.9.3-*.whl')
        self.assertEqual(result['matching_wheels'], 1)
        self.assertFalse(result['wheel_sample']['sample_readable'])

    def test_server_ignoring_range_does_not_cause_a_full_download_or_print_signed_urls(self):
        response = Response(b'PK\x03\x04' + b'content' * 2000)
        response.geturl = lambda: 'https://storage.example.com/file?token=secret'
        with patch('cluster_eval.resource_probe.direct_open', return_value=response):
            result = file_sample('https://pypi.hobot.cc/wheel', wheel=True)
        self.assertEqual(result['sample_bytes'], SAMPLE_LIMIT)
        self.assertTrue(result['sample_readable'])
        self.assertFalse(result['full_download_verified'])
        self.assertNotIn('secret', str(result))

    def test_failure_preserves_other_checks_and_hides_exception_secrets(self):
        with patch('cluster_eval.resource_probe.direct_open', side_effect=OSError('token=secret')):
            result = file_sample('https://example.com')
        self.assertFalse(result['sample_readable'])
        self.assertNotIn('secret', str(result))

    def test_resource_downloads_cannot_run_on_the_development_host(self):
        with patch('cluster_eval.probe.check_execution_context', side_effect=ValueError('not a cluster job')), \
                patch('cluster_eval.resource_probe.direct_open') as connection, \
                self.assertRaises(ValueError):
            resource_inventory()
        connection.assert_not_called()


if __name__ == '__main__':
    unittest.main()
