"""Run the training configurations for a paper figure; preview by default."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .analyze import ROOT, read
from .plot import FIGURES
from .run import expand, load_config


def training_plan(figure='all', seeds=(1, 2, 3)):
    specs = read(ROOT / 'analysis/figure_specs.json')
    figures = FIGURES if figure == 'all' else [figure]
    jobs = {}
    for name in figures:
        if name == 'figure3':
            for panel in specs['narrowness']:
                for variant in panel['variants']:
                    for seed in seeds:
                        run = f"narrowness/{variant['runs'][seed - 1]}"
                        jobs.setdefault(run, {'run': run, 'configs': [variant['configs'][seed - 1]],
                                              'stages': [[0, 1, 2]]})
            continue
        mode = specs['figures'][name]['mode']
        for group_name in specs['figures'][name]['groups']:
            group = specs['groups'][group_name]
            for seed in seeds:
                run = f"{group['family']}/{group['runs'][seed - 1]}"
                job = jobs.setdefault(run, {'run': run, 'depths': set(), 'group': group, 'seed': seed,
                                            'reexposure': False})
                job['depths'].update(group['reverse_depths'] if mode == 'rebound' else group['depths'])
                job['reexposure'] |= mode == 'reexposure'
    plan = []
    for job in jobs.values():
        if 'depths' in job:
            group = job['group']; seed = job['seed']
            depths = sorted(job['depths'])
            configs = [group['configs'][seed - 1][str(depth)] for depth in depths]
            stages = [[0, 1]]
            selections = [configs[0]]
            for depth, config in zip(depths, configs):
                stages.append([2, 3, 4] if job['reexposure'] and depth in group['depths'] else [2])
                selections.append(config)
            job = {'run': job['run'], 'configs': selections, 'stages': stages}
        plan.extend({'run_name': 'paper/' + job['run'], 'config': config, 'stages': stages}
                    for config, stages in zip(job['configs'], job['stages']))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', choices=['all'] + FIGURES, default='all')
    parser.add_argument('--seeds', type=int, choices=[1, 2, 3], nargs='+', default=[1, 2, 3])
    parser.add_argument('--num-processes', type=int, default=4)
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('--seeds must be unique')
    if args.num_processes < 1:
        parser.error('--num-processes must be positive')
    plan = training_plan(args.figure, args.seeds)
    print(json.dumps({'figure': args.figure, 'jobs': plan}, indent=2), flush=True)
    if args.execute:
        missing = set()
        for config in {job['config'] for job in plan}:
            expand(load_config(ROOT / config), missing)
        if missing:
            parser.error('Set model variables before execution: ' + ', '.join(sorted(missing)))
    if args.execute or args.check_data:
        # Validate every input before starting the first expensive run.
        for job in plan:
            command = [sys.executable, '-m', 'alignment_dynamics.run', '--config', job['config'],
                       '--run-name', job['run_name'], '--stages', *map(str, job['stages']),
                       '--num-processes', str(args.num_processes), '--check-data', '--dry-run']
            subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    if args.execute:
        for job in plan:
            command = [sys.executable, '-m', 'alignment_dynamics.run', '--config', job['config'],
                       '--run-name', job['run_name'], '--stages', *map(str, job['stages']),
                       '--num-processes', str(args.num_processes), '--execute']
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
