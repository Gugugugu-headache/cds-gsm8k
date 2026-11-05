#!/usr/bin/env python3
"""Paired GSM8K comparison; no model calls or third-party dependencies."""
import argparse
import math
import random
from pathlib import Path

from cds_pipeline import grade, read, write


def compare(before, after, seed=42, repeats=2000):
    if before['meta']['split'] != 'test' or after['meta']['split'] != 'test':
        raise ValueError('Final comparison requires the test split')
    a, b = before['records'], after['records']
    if len(a) != 1319 or len(b) != len(a):
        raise ValueError('Final comparison requires all 1319 official test items')
    for left, right in zip(a, b):
        if any(left[k] != right[k] for k in ['item_id', 'input', 'output']):
            raise ValueError('Test records are not aligned')
        for row in [left, right]:
            if row['correct'] != grade(row['response'], row['output'])['correct']:
                raise ValueError('Stored correctness is inconsistent')
    if before['meta']['prompt'] != after['meta']['prompt']:
        raise ValueError('Evaluation prompts differ')
    differences = [int(right['correct']) - int(left['correct']) for left, right in zip(a, b)]
    improved, regressed = differences.count(1), differences.count(-1)
    discordant = improved + regressed
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(improved, regressed)+1))
            / 2 ** discordant) if discordant else 1.0
    rng = random.Random(seed)
    estimates = sorted(sum(rng.choices(differences, k=len(a))) / len(a) for _ in range(repeats))
    return {'n': len(a), 'before_accuracy': sum(r['correct'] for r in a) / len(a),
            'after_accuracy': sum(r['correct'] for r in b) / len(b),
            'gain_percentage_points': 100 * sum(differences) / len(a),
            'wrong_to_right': improved, 'right_to_wrong': regressed,
            'paired_bootstrap_95ci_percentage_points': [100 * estimates[int(repeats*.025)],
                                                       100 * estimates[int(repeats*.975)]],
            'mcnemar_exact_two_sided_p': p, 'bootstrap_seed': seed, 'bootstrap_repeats': repeats,
            'limitation': 'CI covers test-item sampling only, not synthesis/training seed variability.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(read(args.before), read(args.after))
    write(args.output, result)
    print(result)


if __name__ == '__main__':
    main()
