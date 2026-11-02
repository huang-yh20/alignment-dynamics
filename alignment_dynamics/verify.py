"""Check released experiment inputs and, optionally, processed-data checksums."""
import argparse
import hashlib
import json
from pathlib import Path

from .analyze import ROOT, read
from .run import data_paths, load_config


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', action='store_true', help='Also verify prepared dataset checksums')
    args = parser.parse_args()
    failures = []
    catalog = read(ROOT / 'experiments/index.json')
    for row in catalog:
        path = ROOT / row['config']
        if not path.is_file():
            failures.append(f'Missing configuration: {path}')
            continue
        config = load_config(path)
        if args.data:
            failures.extend(f'Missing data: {p}' for p in data_paths(config) if not p.is_file())
    checked = 0
    if args.data:
        for row in read(ROOT / 'data_preparation/manifest.json'):
            path = ROOT / row['target']
            checked += 1
            if not path.is_file() or digest(path) != row['sha256']:
                failures.append(f'Data missing or checksum differs: {path}')
    print(json.dumps({'configurations': len(catalog), 'data_checksums': checked,
                      'failures': sorted(set(failures))}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
