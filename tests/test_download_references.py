"""Offline download tests: revision pinning, resumability and file validation."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import download_references as download


def fixture():
    files = {'config.json': b'{"architectures": ["Qwen2ForCausalLM"]}',
             'tokenizer_config.json': b'{"chat_template": "fixture"}',
             'model.safetensors.index.json': json.dumps(
                 {'weight_map': {'a': 'model-00001-of-00001.safetensors'}}).encode(),
             'model-00001-of-00001.safetensors': b'fake weight fixture'}
    sizes = {name: len(body) for name, body in files.items()}
    sizes['consolidated.safetensors'] = 9000
    sizes['original/model.safetensors'] = 9000
    info = SimpleNamespace(sha='a' * 40, siblings=[
        SimpleNamespace(rfilename=name, size=size) for name, size in sizes.items()])
    return files, info


def snapshot_for(files):
    def snapshot(**kwargs):
        for name, body in files.items():
            path = Path(kwargs['local_dir']) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
    return Mock(side_effect=snapshot)


class ReferenceDownloadTests(unittest.TestCase):
    def test_list_is_offline_and_does_not_import_or_call_hub(self):
        out, hub = io.StringIO(), Mock()
        with patch.dict(sys.modules, {'huggingface_hub': hub}), \
             patch.object(sys, 'argv', ['download_references.py', '--list']), \
             contextlib.redirect_stdout(out):
            download.main()
        self.assertEqual(json.loads(out.getvalue()), download.MODELS)
        hub.HfApi.assert_not_called()
        hub.snapshot_download.assert_not_called()

    def test_download_pins_commit_and_avoids_duplicate_weight_formats(self):
        files, info = fixture()
        api, snapshot = Mock(), snapshot_for(files)
        api.model_info.return_value = info
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = download.MODELS['mistral7b_v03']
            report = download.download_one(repo, root, api, snapshot, 2)
            destination = root / repo.split('/')[-1]
            pin = download.read(destination / download.PIN)
            self.assertEqual(pin['revision'], info.sha)
            self.assertEqual(set(pin['files']), set(files))
            self.assertEqual(download.read(destination / 'download_provenance.json'),
                             {'repo_id': repo, 'revision': info.sha})
            args = snapshot.call_args.kwargs
            self.assertIs(args['token'], False)
            self.assertEqual(args['revision'], info.sha)
            self.assertEqual(args['max_workers'], 2)
            self.assertEqual(set(args['allow_patterns']), set(files))
            self.assertEqual(report['files_checked'], len(files))

    def test_interrupted_download_reuses_pin_without_resolving_new_main(self):
        files, info = fixture()
        api = Mock()
        api.model_info.return_value = info
        with tempfile.TemporaryDirectory() as tmp:
            root, repo = Path(tmp), download.MODELS['qwen25_15b']
            with self.assertRaises(TimeoutError):
                download.download_one(repo, root, api, Mock(side_effect=TimeoutError), 1)
            destination = root / repo.split('/')[-1]
            pin_bytes = (destination / download.PIN).read_bytes()
            self.assertFalse((destination / 'download_provenance.json').exists())
            api.model_info.side_effect = AssertionError('Do not resolve main again')
            download.download_one(repo, root, api, snapshot_for(files), 1)
            api.model_info.assert_called_once()
            self.assertEqual((destination / download.PIN).read_bytes(), pin_bytes)

    def test_missing_wrong_size_or_unpinned_shards_cannot_be_marked_ready(self):
        files, info = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'model'
            api = Mock()
            api.model_info.return_value = info
            pin = download.pin_revision(api, download.MODELS['phi35_mini'], destination)
            snapshot_for(files)(local_dir=str(destination))
            shard = destination / 'model-00001-of-00001.safetensors'
            shard.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing or empty'):
                download.validate_download(destination, pin)
            shard.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'size differs'):
                download.validate_download(destination, pin)
            shard.write_bytes(files[shard.name])
            index = destination / 'model.safetensors.index.json'
            index.write_text('{"weight_map":{"a":"absent.safetensors"}}')
            pin['files'][index.name] = index.stat().st_size
            with self.assertRaisesRegex(ValueError, 'Shard missing'):
                download.validate_download(destination, pin)

    def test_existing_unmanaged_or_different_model_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination, api = Path(tmp), Mock()
            original = destination / 'config.json'
            original.write_text('{"existing":true}')
            with self.assertRaisesRegex(ValueError, 'not empty'):
                download.pin_revision(api, download.MODELS['qwen15_4b'], destination)
            self.assertEqual(original.read_text(), '{"existing":true}')
            download.write(destination / download.PIN,
                           {'repo_id': 'other/model', 'revision': 'a' * 40, 'files': {'config.json': 17}})
            with self.assertRaisesRegex(ValueError, 'differs'):
                download.pin_revision(api, download.MODELS['qwen15_4b'], destination)
            api.model_info.assert_not_called()

    def test_repository_paths_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['../outside.json', '/absolute.json', 'a/../../weight.safetensors']:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    download.safe_file(root, name)
            self.assertEqual(download.safe_file(root, 'tokenizer.json'), root / 'tokenizer.json')

    def test_bounded_retries_continue_to_other_models_and_report_failure(self):
        hub = Mock()
        failures = [TimeoutError('offline fixture'), TimeoutError('offline fixture'), {'ok': True}]
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(sys.modules, {'huggingface_hub': hub}), \
             patch.object(download, 'download_one', side_effect=failures) as one, \
             patch.object(sys, 'argv', ['download_references.py', '--models', 'qwen25_05b',
                                       'smollm2_17b', '--root', tmp, '--attempts', '2']), \
             contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as error:
            download.main()
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(one.call_count, 3)
        self.assertEqual(one.call_args.args[0], download.MODELS['smollm2_17b'])
        self.assertIn('Retry these models with --models: qwen25_05b', out.getvalue())
        hub.HfApi.assert_called_once_with(token=False)


if __name__ == '__main__':
    unittest.main()
