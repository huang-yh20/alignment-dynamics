"""Portable launcher; heavyweight training imports happen only in worker processes."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    import yaml
    with open(path, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def expand(value, missing):
    if isinstance(value, dict):
        return {key: expand(item, missing) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, missing) for item in value]
    if isinstance(value, str):
        def substitute(match):
            key = match.group(1)
            if not os.environ.get(key):
                missing.add(key)
                return match.group(0)
            return os.environ[key]
        return re.sub(r'\$\{([A-Z0-9_]+)\}', substitute, value)
    return value


def data_paths(config):
    data = config['data']
    base = ROOT / data['processed_dir'] / f"seed_{data['seed']}"
    return sorted({base / data[key] for key in
                   ['train_positive_data', 'train_negative_data', 'test_positive_data', 'test_negative_data']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Path from experiments/index.json')
    parser.add_argument('--stages', nargs='+', type=int, choices=range(5), default=None)
    parser.add_argument('--run-name', help='Shared name for depth sweeps; outputs go below runs/')
    parser.add_argument('--seed', type=int, help='Training seed (default: historical model.seed)')
    parser.add_argument('--num-processes', type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true', help='Default: show plan without loading models')
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    catalog = json.loads((ROOT / 'experiments/index.json').read_text())
    entry = next((row for row in catalog if (ROOT / row['config']).resolve() == config_path), None)
    if entry is None:
        parser.error('Config must be a cataloged experiment; see experiments/index.json')
    if args.num_processes < 1:
        parser.error('--num-processes must be positive')
    config = load_config(config_path)
    stages = args.stages if args.stages is not None else entry['supported_stages']
    if any(stage not in entry['supported_stages'] for stage in stages):
        parser.error('This configuration only supports stages: ' + ', '.join(map(str, entry['supported_stages'])))
    missing = set()
    config = expand(config, missing)
    seed = args.seed if args.seed is not None else config['model'].get('seed', 42)
    name = args.run_name or f"{entry['family']}/{config_path.parent.name}/{config_path.stem}"
    out = (ROOT / 'runs' / name).resolve()
    if not out.is_relative_to(ROOT / 'runs') or out == ROOT / 'runs':
        parser.error('--run-name must identify a directory below this repository runs/')
    plan = {'family': entry['family'], 'backend': entry['backend'], 'config': entry['config'],
            'stages': stages, 'seed': seed, 'data_seed': config['data']['seed'],
            'num_processes': args.num_processes, 'learning_rate': config['training']['learning_rate'],
            'output_dir': str(out), 'unresolved_model_variables': sorted(missing),
            'data': {str(path.relative_to(ROOT)): path.is_file() for path in data_paths(config)}}
    print(json.dumps(plan, indent=2), flush=True)
    if args.check_data and not all(plan['data'].values()):
        parser.exit(1, 'Missing processed data; use python -m alignment_dynamics.import_data\n')
    if not args.execute:
        return
    if missing:
        parser.error('Set model variables before execution: ' + ', '.join(sorted(missing)))
    if not all(plan['data'].values()):
        parser.error('Processed data missing; import it before execution')
    import yaml
    out.mkdir(parents=True, exist_ok=True)
    # Per-stage configs preserve the effective settings of earlier depth sweeps.
    effective = out / 'effective_configs'
    effective.mkdir(exist_ok=True)
    for stage in stages:
        if stage in [0, 1] and (out / ['stage_0_baseline', 'stage_1_forward'][stage]).exists():
            parser.error(f'Stage {stage} already exists in {out}; use a fresh run name or omit this stage')
        if stage == 2 and not (out / 'stage_1_forward' / f"P{config['saving']['reverse_start_step']}").is_dir():
            parser.error('Run Stage 1 first; required forward checkpoint is missing')
        if stage == 3 and not (out / 'stage_0_baseline/metrics.json').is_file():
            parser.error('Run Stage 0 before score matching')
        path = effective / f"{config_path.parent.name}_{config_path.stem}_stage{stage}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        command = [sys.executable, '-m', 'accelerate.commands.launch',
                   '--config_file', str(ROOT / 'configs/runtime/accelerate_config.yaml'),
                   '--num_processes', str(args.num_processes), '--num_machines', '1',
                   '--mixed_precision', 'bf16', '--dynamo_backend', 'no',
                   '-m', f"alignment_dynamics.backends.{entry['backend']}.experiment",
                   '--config', str(path), '--seed', str(seed), '--stage', str(stage),
                   '--output_dir', str(out)]
        print('Launching:', ' '.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
