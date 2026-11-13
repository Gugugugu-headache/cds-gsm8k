"""Offline Q10 migration and strictly scoped interrupted-augmentation cleanup."""
from contextlib import ExitStack, contextmanager, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cds_pipeline as cp
import ollama_teacher as ot
import qwen15_cds_ollama as job
import qwen15_cds_quality10 as q10
import qwen15_cds_ollama_compat as repeat_fix
import test_qwen15_cds_ollama as source_tests
from test_ollama_teacher import reply, metadata
from test_teacher_sample_format import batch


def save_cache(cfg, run, request, content):
    key = q10.cache_key(cfg, request)
    path = run / 'teacher_cache' / (key + '.json')
    cp.write(path, {
        'request': request, 'url': cfg['teacher']['base_url_default'],
        'content': content, 'finish_reason': 'stop', 'native_response': reply(content),
        'config_hash': cp.digest(cfg), 'model_digest': cfg['teacher']['identity']['digest'],
        'native_request': repeat_fix.ContextRepeatTeacher(cfg, run).native_request(request)})
    return key, path


@contextmanager
def fixture(root):
    case = source_tests.LocalCdsTests()
    original = case.fixture(root)
    with case.context(root, original), ExitStack() as stack:
        cfg, run = job.prepare()
        job.profile()
        rows = []
        for index, score in enumerate([10, 10, 9, 8, 7, None]):
            row = {'item_id': f'synthesis:{index}', 'instruction': cp.SYN_INSTRUCTION,
                   'input': f'A hypothetical shop has {100 + index} items. How many items?',
                   'output': f'There are {100 + index} items.', 'label': ['Addition'],
                   'provenance': {'stage': 'fine', 'request': cp.digest(['synthesis', index])}}
            rows.append(row)
        cp.write(run / 'synthesis.json', rows)
        cp.write(run / 'diagnostics.json', [{'item_id': 'diag:2556', 'fallback': False}])
        accepted, rejected = [], []
        for row, score in zip(rows, [10, 10, 9, 8, 7, None]):
            prompt = cp.prompt_module().prompt4filter.format(
                question=row['input'], answer=row['output'], label=row['label'])
            request = q10.payload(cfg, [cp.user_message(prompt)], scoring=True)
            content = f'Correctness assessment.\nScore: {score}' if score is not None else 'No scalar returned.'
            key, _ = save_cache(cfg, run, request, content)
            rated = {**row, 'quality_score': score, 'quality_request': key}
            (accepted if score is not None and score >= 8 else rejected).append(rated)
        cp.write(run / 'seed_quality.json', accepted)
        cp.write(run / 'seed_quality.rejected.json', rejected)
        stack.enter_context(patch.object(q10, 'SOURCE_CONFIG', job.CONFIG_PATH))
        stack.enter_context(patch.object(q10, 'SOURCE_RUN', job.RUN_DIR))
        stack.enter_context(patch.object(q10, 'CONFIG_PATH', root / 'src/configs/q10.json'))
        stack.enter_context(patch.object(q10, 'RUN_DIR', str(root / 'new_q10_run')))
        yield cfg, run, rows


class Quality10Tests(unittest.TestCase):
    def test_prepare_reuses_scores_without_network_and_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as tmp, fixture(Path(tmp)) as (old_cfg, old, samples):
            before = {p: p.read_bytes() for p in Path(tmp).rglob('*') if p.is_file()}
            with patch.object(ot, 'rpc') as rpc:
                q10.prepare()
                cfg, run, _ = q10.load()
                q10.prepare()
                rpc.assert_not_called()
            self.assertEqual(cfg['quality_min_score'], 10)
            changed = {key for key in cfg if cfg[key] != old_cfg[key]}
            self.assertEqual(changed, {'run_dir', 'quality_min_score', 'allocation_entrypoint'})
            self.assertEqual(cp.read(run / 'synthesis.json'), samples)
            kept = cp.read(run / 'seed_quality.json')
            self.assertEqual([r['item_id'] for r in kept], ['synthesis:0', 'synthesis:1'])
            self.assertEqual(len(cp.read(run / 'seed_quality.rejected.json')), 4)
            self.assertEqual(cp.read(run / 'data/diag.json'), cp.read(old / 'data/diag.json'))
            self.assertEqual(cp.read(run / 'eval/raw_diag.json')['records'],
                             cp.read(old / 'eval/raw_diag.json')['records'])
            self.assertFalse((run / 'teacher_cache').exists())
            self.assertEqual(cp.read(run / 'quality10_migration.json')['augmentation_requests'], 1)
            for path, data in before.items():
                self.assertEqual(path.read_bytes(), data)

    def test_score_or_sample_tampering_blocks_prepare_before_new_artifacts(self):
        for mode in ['score', 'question', 'cache']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp, \
                 fixture(Path(tmp)) as (_, old, _):
                path = old / 'seed_quality.json'
                rows = cp.read(path)
                if mode == 'score':
                    rows[0]['quality_score'] = 9
                elif mode == 'question':
                    rows[0]['input'] = 'Changed question'
                else:
                    cache = old / 'teacher_cache' / (rows[0]['quality_request'] + '.json')
                    data = cp.read(cache)
                    data['content'] = 'Score: 8'
                    cache.write_text(cp.json.dumps(data))
                if mode != 'cache':
                    path.write_text(cp.json.dumps(rows))
                with patch.object(ot, 'rpc') as rpc, self.assertRaises(ValueError):
                    q10.prepare()
                rpc.assert_not_called()
                self.assertFalse(cp.resolve(q10.RUN_DIR).exists())

    def test_same_directory_or_changed_migration_refused(self):
        with tempfile.TemporaryDirectory() as tmp, fixture(Path(tmp)) as (_, old, _):
            with patch.object(q10, 'RUN_DIR', str(old)), self.assertRaises(ValueError):
                q10.prepare()
            q10.prepare()
            path = cp.resolve(q10.RUN_DIR) / 'quality10_migration.json'
            data = cp.read(path)
            data['seed_accepted'] = 999
            path.write_text(cp.json.dumps(data))
            with self.assertRaises(ValueError):
                q10.load()

    def test_cleanup_only_two_reported_augmentation_caches_and_audits(self):
        with tempfile.TemporaryDirectory() as tmp, fixture(Path(tmp)) as (cfg, old, _):
            requests = q10.first_augmentation_requests(cfg, cp.read(old / 'seed_quality.json'))
            original = {p: p.read_bytes() for p in old.rglob('*') if p.is_file()}
            removed = []
            for key, request in requests.items():
                _, path = save_cache(cfg, old, request, batch())
                audit = old / 'teacher_repeat_scope_requests' / (key + '.json')
                cp.write(audit, {'request': key})
                parser_audit = old / 'teacher_sample_parsing' / (key + '.fixture.json')
                cp.write(parser_audit, {'request': key, 'stage': 'rewrite'})
                removed += [path, audit, parser_audit]
            unrelated = old / 'teacher_cache' / ('f' * 64 + '.json')
            cp.write(unrelated, {'leave': 'unchanged'})
            with patch.object(q10, 'INTERRUPTED_KEYS', set(requests)), patch.object(ot, 'rpc') as rpc:
                q10.discard_interrupted()
                q10.discard_interrupted()
                rpc.assert_not_called()
            self.assertTrue(all(not p.exists() for p in removed))
            self.assertEqual(cp.read(unrelated), {'leave': 'unchanged'})
            for path, data in original.items():
                self.assertEqual(path.read_bytes(), data)
            q10.prepare()  # Deleting only augmentation does not invalidate source-score reuse.

    def test_cleanup_missing_second_cache_is_safe_and_wrong_keys_or_full_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, fixture(Path(tmp)) as (cfg, old, _):
            requests = q10.first_augmentation_requests(cfg, cp.read(old / 'seed_quality.json'))
            key, request = next(iter(requests.items()))
            _, path = save_cache(cfg, old, request, batch())
            raw = path.read_bytes()
            with patch.object(q10, 'INTERRUPTED_KEYS', {'wrong'}), self.assertRaises(ValueError):
                q10.discard_interrupted()
            self.assertEqual(path.read_bytes(), raw)
            cp.write(old / 'augmented.json', [])
            with patch.object(q10, 'INTERRUPTED_KEYS', set(requests)), self.assertRaises(ValueError):
                q10.discard_interrupted()
            self.assertEqual(path.read_bytes(), raw)
            (old / 'augmented.json').unlink()
            with patch.object(q10, 'INTERRUPTED_KEYS', set(requests)):
                q10.discard_interrupted()
            self.assertFalse(path.exists())

    def test_q10_cli_augmentation_quality_and_selection_use_ten_and_restore_routes(self):
        with tempfile.TemporaryDirectory() as tmp, fixture(Path(tmp)) as (_, old, _):
            q10.prepare()
            old_load, old_config, old_run = job.load, job.CONFIG_PATH, job.RUN_DIR
            chats = []

            def fake(base, route, payload=None, timeout=30):
                if route != '/api/chat':
                    return metadata(base, route, payload, timeout)
                chats.append(payload)
                return reply('Assessment.\nScore: 9' if payload['options']['temperature'] == 0 else batch())

            before = {p: p.read_bytes() for p in old.rglob('*') if p.is_file()}
            with patch.object(ot, 'rpc', side_effect=fake):
                for command in [['augment'], ['quality', '--pool', 'augmented'], ['select']]:
                    with patch.object(sys, 'argv', ['quality10.py', *command]):
                        q10.main()
            run = cp.resolve(q10.RUN_DIR)
            self.assertEqual(len(cp.read(run / 'augmented.json')), 3)
            self.assertEqual(cp.read(run / 'augmented_quality.json'), [])
            self.assertEqual(len(cp.read(run / 'augmented_quality.rejected.json')), 3)
            self.assertTrue(cp.read(run / 'train_cds.json'))
            self.assertEqual(len(chats), 4)
            for path, data in before.items():
                self.assertEqual(path.read_bytes(), data)
            self.assertIs(job.load, old_load)
            self.assertEqual((job.CONFIG_PATH, job.RUN_DIR), (old_config, old_run))


if __name__ == '__main__':
    unittest.main()
