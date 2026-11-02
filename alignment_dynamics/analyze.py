"""Export checkpoint metrics and forward-domain likelihood statistics."""
import argparse
import csv
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def metric(metrics, key):
    if key in metrics:
        return metrics[key]
    if key.startswith('minus_') and key[6:] in metrics:
        return -metrics[key[6:]]
    return None


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def export(root, output):
    rows = []
    for path in sorted(root.glob('*/*/*/*/metadata.json')):
        family, experiment, stage, checkpoint, _ = path.relative_to(root).parts
        data = read(path)
        config_path = root / family / experiment / 'config.json'
        config = read(config_path) if config_path.is_file() else {}
        seed = config.get('data', {}).get('seed', 1)
        for name, value in (data.get('metrics') or {}).items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                rows.append({'family': family, 'experiment': experiment, 'seed': seed,
                             'stage': stage, 'checkpoint': checkpoint, 'step': data.get('step'),
                             'metric': name, 'value': value, 'source': str(path.relative_to(root))})
    write_csv(output / 'checkpoint_metrics.csv', rows,
              ['family', 'experiment', 'seed', 'stage', 'checkpoint', 'step', 'metric', 'value', 'source'])
    return {'metric_values': len(rows), 'output': str(output / 'checkpoint_metrics.csv')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-root', type=Path, default=ROOT / 'artifacts/metrics')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/tables')
    args = parser.parse_args()
    print(json.dumps(export(args.metrics_root, args.output), indent=2))


if __name__ == '__main__':
    main()
