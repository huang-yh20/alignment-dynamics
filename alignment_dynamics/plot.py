"""Plot Figures 2–6 from checkpoint metrics using the paper's panel layout."""
import argparse
import json
from pathlib import Path
import re

from .analyze import ROOT, read, write_csv

FIGURES = ['figure2', 'figure3', 'figure4', 'figure5', 'figure6']


def load_score(path, metric):
    if not path.is_file():
        return float('nan')
    return read(path).get('metrics', {}).get(metric, float('nan'))


def recovery(run, depth):
    matches = sorted((run / 'stage_3_recovery').glob(f'best_match_P{depth}_N*'))
    if len(matches) > 1:
        raise ValueError(f'Ambiguous recovery checkpoint: {run}, depth {depth}')
    return matches[0] if matches else None


def group_curves(root, group, mode, seeds=(1, 2, 3), paper_single=False):
    """Return explicitly ordered checkpoint samples, without interpolation."""
    import numpy as np
    names = [group['runs'][seed - 1] for seed in seeds]
    if paper_single:
        names = [group['paper_run']]
    runs = [root / group['family'] / name for name in names]
    key = group['metric']
    baseline = np.array([load_score(run / 'stage_0_baseline/P0/metadata.json', key) for run in runs])
    curves = []
    if mode == 'rebound':
        forward = [0] + group['forward_steps']
        values = [[load_score(run / ('stage_0_baseline/P0' if step == 0 else f'stage_1_forward/P{step}') / 'metadata.json', key)
                   for step in forward] for run in runs]
        for depth in group['reverse_depths']:
            steps = [0] + group['reverse_steps']
            rows = [[load_score(run / (f'stage_1_forward/P{depth}' if step == 0 else f'stage_2_reverse/P{depth}-N{step}') / 'metadata.json', key)
                     for step in steps] for run in runs]
            curves.append((str(depth), np.array(steps) + depth, np.array(rows)))
        curves.append(('Forward', np.array(forward), np.array(values)))
    else:
        steps = [0] + group['reexposure_steps']
        for depth in [0] + group['depths']:
            rows = []
            for run in runs:
                if depth == 0:
                    paths = [run / 'stage_0_baseline/P0'] + [run / f'stage_1_forward/P{step}' for step in steps[1:]]
                else:
                    found = recovery(run, depth)
                    prefix = found.name.replace('best_match_', '').replace('_', '-') + '-P' if found else None
                    paths = [found] + [run / 'stage_4_forward' / f'{prefix}{step}' if found else None for step in steps[1:]]
                rows.append([load_score(path / 'metadata.json', key) if path else np.nan for path in paths])
            curves.append((str(depth), np.array(steps), np.array(rows)))
    return curves, baseline, names


def statistics(values):
    """Use the available seeds at each point, as in the multi-seed plot scripts."""
    import numpy as np
    count = np.isfinite(values).sum(axis=0)
    mean = np.full(values.shape[1], np.nan)
    std = np.full(values.shape[1], np.nan)
    valid = count > 0
    mean[valid] = np.nansum(values[:, valid], axis=0) / count[valid]
    std[valid] = np.sqrt(np.nansum((values[:, valid] - mean[valid]) ** 2, axis=0) / count[valid])
    return mean, std, count


def render(figure, root, output, specs, seeds=(1, 2, 3), paper_single=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.titlesize': 11, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'axes.linewidth': .7, 'lines.linewidth': 1.1,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    output.mkdir(parents=True, exist_ok=True)
    points = []

    def curve(ax, x, values, color, label, panel, errorbar=False):
        mean, std, count = statistics(values)
        if errorbar and values.shape[0] > 1:
            ax.errorbar(x, mean, yerr=std, color=color, marker='o', markersize=3,
                        capsize=2, elinewidth=.7, label=label, alpha=.8)
        else:
            ax.plot(x, mean, color=color, marker='o', markersize=3, label=label)
        if values.shape[0] > 1:
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=.15, linewidth=0)
        for j, step in enumerate(x):
            row = {'figure': figure, 'panel': panel, 'curve': label, 'step': int(step),
                   'seeds': int(count[j]), 'expected_seeds': values.shape[0],
                   'mean': float(mean[j]) if count[j] else '', 'std': float(std[j]) if count[j] else ''}
            row.update({f'seed_{seed}': float(values[i, j]) if np.isfinite(values[i, j]) else '' for i, seed in enumerate([1] if paper_single else seeds)})
            points.append(row)

    def draw(ax, name, mode):
        group = specs['groups'][name]
        curves, baseline, _ = group_curves(root, group, mode, seeds, paper_single)
        colored = [c for c in curves if c[0] != 'Forward']
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(colored)))
        idx = 0
        for label, x, values in curves:
            color = 'black' if label == 'Forward' else colors[idx]
            idx += label != 'Forward'
            curve(ax, x, values, color, label, name)
        finite = baseline[np.isfinite(baseline)]
        if len(finite):
            ax.axhline(float(finite.mean()), color='gray', linestyle='--', linewidth=1, label='Baseline')
        ax.set_title(group['model_title'], pad=7)
        ax.set_xlabel('Number of Steps')
        ax.set_ylabel({'safety_score': 'Safety Rate', 'positive_score': 'Positive Score',
                       'obvious_misalignment_rate': 'Misalignment Rate'}[group['metric']])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if mode != 'rebound':
            ax.set_xticks([0, 64, 128, 192, 256])
            ax.grid(True, alpha=.3, linewidth=.5)
        return ax.get_legend_handles_labels()

    def labels(fig, x, y, letter, title=None, vertical=False):
        fig.text(x, y, letter, fontsize=14, fontweight='bold', va='top')
        if title:
            fig.text(x + .018, y - .075 if vertical else y - .015, title,
                     rotation=90 if vertical else 0, fontsize=11,
                     ha='left', va='top')

    def legend(fig, ax, loc, bbox, ncol=1, title='Forward Steps'):
        handles, names = ax.get_legend_handles_labels()
        fig.legend(handles, names, loc=loc, bbox_to_anchor=bbox, ncol=ncol,
                   fontsize=8, title=title, title_fontsize=9,
                   columnspacing=1.1, handlelength=1.8)

    if figure == 'figure2':
        fig, axes = plt.subplots(1, 4, figsize=(11.3, 2.85))
        fig.subplots_adjust(left=.055, right=.865, bottom=.22, top=.81, wspace=.55)
        for ax, name in zip(axes, specs['figures'][figure]['groups']):
            draw(ax, name, 'rebound')
        labels(fig, .005, .96, 'a')
        labels(fig, .235, .96, 'b')
        legend(fig, axes[-1], 'center left', (.875, .48))
    elif figure == 'figure4':
        fig = plt.figure(figsize=(11.9, 6.6))
        axes = np.array([[fig.add_axes([x, y, .172, .31])
                          for x in [.092, .309, .604, .817]] for y in [.625, .165]])
        for ax, name in zip(axes.flat, specs['figures'][figure]['groups']):
            draw(ax, name, 'reexposure')
        for x, y, letter, title in [(.005,.97,'a','IMDb (Stage 1 Positive)'),(.515,.97,'b','IMDb (Stage 1 Negative)'),
                                    (.005,.51,'c','BeaverTails'),(.515,.51,'d','Misalignment')]:
            labels(fig, x, y, letter, title, vertical=True)
        legend(fig, axes[0, 0], 'lower center', (.53, .01), ncol=6, title='Stage 1 Forward Steps')
    elif figure == 'figure5':
        fig, axes = plt.subplots(3, 3, figsize=(11.9, 11.35))
        fig.subplots_adjust(left=.105, right=.84, bottom=.07, top=.92, hspace=.52, wspace=.4)
        groups = specs['figures'][figure]['groups']
        for ax, name in zip(axes.flat, groups):
            draw(ax, name, 'rebound')
        axes[2, 0].set_title('')
        axes[2, 1].set_visible(False)
        axes[2, 2].set_visible(False)
        for y, letter, title in [(.97,'a','IMDb (Stage 1 Positive)'),(.65,'b','IMDb (Stage 1 Negative)'),(.33,'c','BeaverTails')]:
            labels(fig, .025, y, letter, title, vertical=True)
        legend(fig, axes[0, 2], 'center left', (.85, .8))
        legend(fig, axes[1, 2], 'center left', (.85, .485))
        legend(fig, axes[2, 0], 'center left', (.35, .17))
    elif figure == 'figure6':
        fig, axes = plt.subplots(1, 3, figsize=(11.9, 3.85))
        fig.subplots_adjust(left=.085, right=.855, bottom=.18, top=.8, wspace=.35)
        for ax, name, title in zip(axes, specs['figures'][figure]['groups'],
                                   ['IMDb (Stage 1 Positive)', 'IMDb (Stage 1 Negative)', 'Emergent Misalignment']):
            draw(ax, name, 'reexposure')
            ax.set_title(title, pad=16)
        legend(fig, axes[-1], 'center left', (.865, .48))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(11.9, 6.65))
        fig.subplots_adjust(left=.10, right=.985, bottom=.17, top=.9, wspace=.28, hspace=.4)
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, 4))
        for ax, panel in zip(axes.flat, specs['narrowness']):
            baselines = []
            for color, variant in zip(colors, panel['variants']):
                rows = []
                for seed in ([1] if paper_single else seeds):
                    run = root / 'narrowness' / variant['runs'][seed - 1]
                    if panel['mode'] == 'forward':
                        steps = [0, 300, 600, 900]
                        paths = [run / 'stage_0_baseline/P0'] + [run / f'stage_1_forward/P{step}' for step in steps[1:]]
                    else:
                        steps = [0, 16, 32, 64, 128]
                        paths = [run / 'stage_1_forward/P900'] + [run / f'stage_2_reverse/P900-N{step}' for step in steps[1:]]
                    rows.append([load_score(path / 'metadata.json', 'safety_score') for path in paths])
                    baselines.append(load_score(run / 'stage_0_baseline/P0/metadata.json', 'safety_score'))
                curve(ax, steps, np.array(rows), color, variant['label'], panel['title'], errorbar=True)
            finite = np.array(baselines); finite = finite[np.isfinite(finite)]
            if len(finite):
                ax.axhline(float(finite.mean()), color='gray', linestyle='--', linewidth=1, alpha=.7, label='Baseline')
            ax.set_xlabel('Number of Steps')
            ax.set_ylabel('Safety Rate')
            ax.set_xticks([0, 300, 600, 900] if panel['mode'] == 'forward' else [0, 30, 60, 90, 120])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.grid(True, alpha=.3, linestyle='--', linewidth=.5)
        for ax, title in zip(axes[0], ['Stage 1 Forward', 'Stage 2 Reverse', 'Stage 2 Reverse (Alpaca)']):
            ax.set_title(title, pad=15)
        for y, letter, model in [(.96, 'a', 'Llama3.1-8B'), (.51, 'b', 'Gemma-2-2B')]:
            labels(fig, .015, y, letter, model, vertical=True)
        handles, names = axes[0, 0].get_legend_handles_labels()
        ordered = ['Baseline'] + [v['label'] for v in specs['narrowness'][0]['variants']]
        lookup = dict(zip(names, handles))
        fig.legend([lookup[name] for name in ordered], ordered, loc='lower center',
                   bbox_to_anchor=(.54, .01), ncol=5, fontsize=8, columnspacing=1.2)
    for ext in ['svg', 'pdf', 'png']:
        fig.savefig(output / f'{figure}.{ext}', dpi=180)
    plt.close(fig)
    fields = ['figure', 'panel', 'curve', 'step', 'seeds', 'expected_seeds', 'mean', 'std'] + [f'seed_{seed}' for seed in ([1] if paper_single else seeds)]
    write_csv(output / f'{figure}_points.csv', points, fields)
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', choices=['all'] + FIGURES, default='all')
    parser.add_argument('--metrics-root', type=Path, default=ROOT / 'artifacts/metrics')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/figures')
    parser.add_argument('--seeds', type=int, choices=[1, 2, 3], nargs='+', default=[1, 2, 3])
    parser.add_argument('--paper-single-seed', action='store_true', help='Plot the individual runs used in the arXiv figures')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('--seeds must be unique')
    specs = read(ROOT / 'analysis/figure_specs.json')
    rows = []
    for figure in FIGURES if args.figure == 'all' else [args.figure]:
        rows.extend(render(figure, args.metrics_root, args.output, specs, args.seeds, args.paper_single_seed))
    print(json.dumps({'curve_points': len(rows), 'incomplete_points': sum(row['seeds'] != row['expected_seeds'] for row in rows),
                      'seeds': args.seeds if not args.paper_single_seed else [1], 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
