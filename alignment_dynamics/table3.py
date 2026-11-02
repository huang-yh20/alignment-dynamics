"""Export the Gemma likelihood measurements used for Table 3."""
import argparse
import json
from pathlib import Path
from .analyze import ROOT, read, write_csv
from .plot import recovery

TASKS = {
    'imdb_positive': ('imdb', 'gemma-2-2b_imdb', 'positive_token_log_prob'),
    'imdb_negative': ('imdb', 'gemma-2-2b_imdb_minus', 'negative_token_log_prob'),
    'safety': ('safety', 'gemma-2-2b_beavertails_qwen_lr5e-7', 'positive_token_log_prob'),
    'emergent': ('emergent', 'gemma-2-2b_misalignment_lr2e-7', 'positive_token_log_prob'),
}


def export(metrics_root, output):
    rows = []
    for task, (family, base, key) in TASKS.items():
        for seed in [1, 2, 3]:
            name = base + (f'_seed{seed}' if seed > 1 or family == 'emergent' else '')
            run = metrics_root / family / name
            for depth in [0, 128, 256, 512, 1024]:
                checkpoint = run / 'stage_0_baseline/P0' if depth == 0 else recovery(run, depth)
                if checkpoint is None:
                    raise FileNotFoundError(f'Missing recovery checkpoint: {run}, depth {depth}')
                path = checkpoint / 'metadata.json'
                value = read(path)['metrics'][key]
                rows.append({'task': task, 'seed': seed, 'depth': depth, 'metric': key,
                             'log_probability': value, 'source': str(path.relative_to(metrics_root))})
    write_csv(output, rows, ['task', 'seed', 'depth', 'metric', 'log_probability', 'source'])
    return {'rows': len(rows), 'output': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-root', type=Path, default=ROOT / 'artifacts/metrics')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/tables/table3.csv')
    args = parser.parse_args()
    print(json.dumps(export(args.metrics_root, args.output), indent=2))


if __name__ == '__main__':
    main()
