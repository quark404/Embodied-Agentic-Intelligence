import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cluster_eval.model_mirror import ROOT, download_file, download_model, validate_lock


class Response(io.BytesIO):
    status = 200
    headers = {}


class ModelMirrorTests(unittest.TestCase):
    def test_committed_manifest_preserves_all_original_digest_identities(self):
        lock = json.loads((ROOT / 'configs/qwen35-modelscope-lock.json').read_text())
        files = validate_lock(lock, 'Qwen/Qwen3.5-27B', 'fc05daec18b0a78c049392ed2e771dde82bdf654')
        self.assertEqual(len([item for item in files if item['path'].endswith('.safetensors')]), 11)
        altered = json.loads(json.dumps(lock))
        altered['files'][0].pop('hf_git_blob_sha1')
        with self.assertRaises(ValueError):
            validate_lock(altered, lock['model'], lock['hf_revision'])

    def test_files_are_published_only_after_hash_and_original_git_blob_validation(self):
        content = b'{"model":"test"}'
        item = {'path': 'config.json', 'size': len(content), 'mirror_revision': 'a' * 40,
                'sha256': hashlib.sha256(content).hexdigest(),
                'hf_git_blob_sha1': hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()}
        for valid in (True, False):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as directory:
                snapshot = Path(directory)
                entry = item if valid else {**item, 'hf_git_blob_sha1': '0' * 40}
                opener = SimpleNamespace(open=lambda *args, **kwargs: Response(content))
                with patch('cluster_eval.model_mirror.build_opener', return_value=opener):
                    if valid:
                        report = download_file(entry, snapshot, 'Qwen/Qwen3.5-27B')
                        self.assertTrue(report['original_digest_verified'])
                        self.assertEqual((snapshot / 'config.json').read_bytes(), content)
                    else:
                        with self.assertRaises(ValueError):
                            download_file(entry, snapshot, 'Qwen/Qwen3.5-27B')
                        self.assertFalse((snapshot / 'config.json').exists())

    def test_no_model_download_is_allowed_on_the_development_host(self):
        with patch('cluster_eval.model_mirror.validate_local_destination', side_effect=ValueError('not allocated')), \
                patch('cluster_eval.model_mirror.build_opener') as opener, self.assertRaises(ValueError):
            download_model(Path('/tmp/cache'), model='Qwen/Qwen3.5-27B', revision='a' * 40)
        opener.assert_not_called()

    def test_bad_sizes_checksums_and_paths_are_not_published(self):
        item = {'path': 'weights.safetensors', 'size': 3, 'mirror_revision': 'a' * 40,
                'sha256': hashlib.sha256(b'abc').hexdigest()}
        for content in (b'ab', b'abcd', b'bad'):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                opener = SimpleNamespace(open=lambda *args, **kwargs: Response(content))
                with patch('cluster_eval.model_mirror.build_opener', return_value=opener), self.assertRaises(ValueError):
                    download_file(item, Path(directory), 'Qwen/Qwen3.5-27B')
                self.assertFalse((Path(directory) / item['path']).exists())
        lock = json.loads((ROOT / 'configs/qwen35-modelscope-lock.json').read_text())
        lock['files'][0]['path'] = '../other-user/model'
        with self.assertRaises(ValueError):
            validate_lock(lock, lock['model'], lock['hf_revision'])


if __name__ == '__main__':
    unittest.main()
