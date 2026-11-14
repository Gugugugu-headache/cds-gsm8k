#!/usr/bin/env python3
"""Forward the frozen Ollama/repeat-fix workflow with versioned sample parsing."""
import argparse
from contextlib import contextmanager
import json
import re
import sys

import cds_pipeline as cp
import qwen15_cds_ollama as job
import qwen15_cds_ollama_compat as repeat_fix
import teacher_sample_descriptive_footer as previous
import teacher_sample_format as headings
import teacher_sample_difficulty_heading as formatting


@contextmanager
def activate_parser():
    original = previous.sample_format_compatibility
    previous.sample_format_compatibility = formatting.sample_format_compatibility
    try:
        yield
    finally:
        previous.sample_format_compatibility = original


def preview(key, count, run_dir=None):
    if not re.fullmatch(r'[0-9a-f]{64}', key or '') or count <= 0:
        raise ValueError('Use a 64-character request hash and positive count.')
    cfg = cp.read(job.CONFIG_PATH if run_dir is None else cp.resolve(run_dir) / 'config.frozen.json')
    run = cp.resolve(cfg['run_dir'])
    if run_dir is not None and run != cp.resolve(run_dir):
        raise ValueError('Preview directory differs from its frozen run_dir.')
    cached = cp.read(run / 'teacher_cache' / (key + '.json'))
    headings.validate_cached_text(run, key, cached['content'])
    rows, mode = formatting.parse_compatible(cached['content'], count, [],
                                             {'request': key}, cp.parse_samples)
    print(json.dumps({'request': key, 'parsed_count': len(rows), 'parser': mode,
                      'preview_only': True, 'questions': [r['input'] for r in rows]},
                     ensure_ascii=False, indent=2))


def main():
    if sys.argv[1:2] == ['parse-cache']:
        parser = argparse.ArgumentParser(description='Offline cache parsing; no model calls or artifact writes.')
        parser.add_argument('command', choices=['parse-cache'])
        parser.add_argument('--request-key', required=True)
        parser.add_argument('--count', type=int, required=True)
        parser.add_argument('--run', help='Explicit run directory for offline preview, including Q10 caches.')
        args = parser.parse_args()
        preview(args.request_key, args.count, run_dir=args.run)
    elif sys.argv[1:2] in [['synthesize'], ['augment']]:
        with activate_parser():
            repeat_fix.main()
    else:
        repeat_fix.main()


if __name__ == '__main__':
    main()
