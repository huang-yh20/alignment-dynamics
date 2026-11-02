import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from alignment_dynamics.analyze import ROOT, export, metric, read
from alignment_dynamics.plot import group_curves, recovery, render, statistics
from alignment_dynamics.reproduce import training_plan
from alignment_dynamics.run import load_config


class ArtifactTests(unittest.TestCase):
    def test_signed_metric_missing_is_not_zero(self):
        self.assertEqual(metric({'positive_score': .6}, 'minus_positive_score'), -.6)
        self.assertIsNone(metric({}, 'safety_score'))

    def test_export_keeps_observations_and_omits_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            experiment = root / 'metrics/imdb/toy_seed2'
            path = experiment / 'stage_0_baseline/P0/metadata.json'
            path.parent.mkdir(parents=True)
            (experiment / 'config.json').write_text(json.dumps({'data': {'seed': 2}}))
            path.write_text(json.dumps({'step': 0, 'metrics': {'positive_score': .52, 'unused': float('nan')}}))
            summary = export(root / 'metrics', root / 'tables')
            self.assertEqual(summary['metric_values'], 1)
            self.assertIn('0.52', (root / 'tables/checkpoint_metrics.csv').read_text())

    def test_statistics_match_original_plotting_reference(self):
        specs = read(ROOT / 'analysis/figure_specs.json')
        cache = {}
        for ref in read(ROOT / 'tests/fixtures/plot_statistics.json'):
            key = (ref['group'], ref['mode'])
            if key not in cache:
                curves, _, _ = group_curves(ROOT / 'artifacts/metrics', specs['groups'][key[0]], key[1])
                cache[key] = {label: (steps, values) for label, steps, values in curves}
            steps, values = cache[key][ref['curve']]
            mean, std, count = statistics(values)
            np.testing.assert_array_equal(steps, ref['steps'])
            np.testing.assert_array_equal(count, ref['count'])
            np.testing.assert_allclose(mean, [np.nan if v is None else v for v in ref['mean']], atol=1e-12)
            np.testing.assert_allclose(std, [np.nan if v is None else v for v in ref['std']], atol=1e-12)

    def test_missing_seed_uses_available_values_without_interpolation(self):
        values = np.array([[.1, np.nan, np.nan], [.3, .6, np.nan], [.5, .8, np.nan]])
        mean, std, count = statistics(values)
        np.testing.assert_allclose(mean[:2], [.3, .7])
        np.testing.assert_allclose(std[:2], [np.std([.1, .3, .5]), .1])
        np.testing.assert_array_equal(count, [3, 2, 0])
        self.assertTrue(np.isnan(mean[2]))

    def test_paper_panel_selection(self):
        figures = read(ROOT / 'analysis/figure_specs.json')['figures']
        self.assertEqual(figures['figure2']['groups'], [
            'llama3-8b_safety', 'llama3-8b_emergent', 'qwen3-8b_emergent', 'gemma-2-2b_emergent'])
        self.assertEqual(figures['figure5']['groups'][-1], 'gemma-2-2b_safety')
        self.assertEqual(len(figures['figure5']['groups']), 7)

    def test_response_diversity_matches_original_notebook_statistics(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = render('figure3', ROOT / 'artifacts/metrics', Path(folder),
                          read(ROOT / 'analysis/figure_specs.json'))
            indexed = {(r['panel'], r['curve'], r['step']): r for r in rows}
            for ref in read(ROOT / 'tests/fixtures/narrowness_statistics.json'):
                for step, mean, std in zip(ref['steps'], ref['mean'], ref['std']):
                    row = indexed[(ref['panel'], ref['curve'], step)]
                    self.assertAlmostEqual(row['mean'], mean, places=12)
                    self.assertAlmostEqual(row['std'], std, places=12)

    def test_figure_seed_identity_is_explicit_and_unique(self):
        specs = read(ROOT / 'analysis/figure_specs.json')
        groups = list(specs['groups'].values())
        groups += [{'family': 'narrowness', 'runs': variant['runs']}
                   for panel in specs['narrowness'] for variant in panel['variants']]
        for group in groups:
            seeds = [read(ROOT / 'artifacts/metrics' / group['family'] / run / 'config.json')['data']['seed']
                     for run in group['runs']]
            self.assertEqual(seeds, [1, 2, 3], group)

    def test_recovery_lookup_does_not_confuse_exposure_depths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'stage_3_recovery/best_match_P128_N5').mkdir(parents=True)
            self.assertIsNone(recovery(root, 12))
            self.assertEqual(recovery(root, 128).name, 'best_match_P128_N5')
            (root / 'stage_3_recovery/best_match_P128_N6').mkdir()
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                recovery(root, 128)

    def test_training_plan_has_one_forward_stage_per_run_and_valid_inputs(self):
        jobs = training_plan()
        baselines = {}
        catalog = {row['config'] for row in read(ROOT / 'experiments/index.json')}
        for job in jobs:
            self.assertIn(job['config'], catalog)
            self.assertNotIn('/dpo/', job['config'])
            self.assertNotIn('/instruct/', job['config'])
            config = load_config(ROOT / job['config'])
            if 0 in job['stages']:
                baselines[job['run_name']] = baselines.get(job['run_name'], 0) + 1
            seed = config['data']['seed']
            self.assertIn(seed, [1, 2, 3])
        self.assertTrue(all(value == 1 for value in baselines.values()))
        self.assertEqual(set(baselines), {job['run_name'] for job in jobs})

    def test_response_diversity_plan_needs_only_stages_zero_through_two(self):
        jobs = training_plan('figure3')
        self.assertEqual(len(jobs), 48)
        self.assertTrue(all(job['stages'] == [0, 1, 2] for job in jobs))

    def test_dry_run_loads_no_training_runtime(self):
        code = """
import sys
from alignment_dynamics.run import main
sys.argv = ['run', '--config', 'experiments/imdb/configs/gemma-2-2b_imdb/reverse128.yaml', '--dry-run']
main()
assert not any(name in sys.modules for name in ['torch', 'trl', 'transformers', 'accelerate'])
"""
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['learning_rate'], '1e-6')

    def test_output_is_isolated_under_runs(self):
        result = subprocess.run([sys.executable, '-m', 'alignment_dynamics.run', '--config',
            'experiments/imdb/configs/gemma-2-2b_imdb/reverse128.yaml', '--run-name', '../../outside', '--dry-run'],
            cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('below this repository', result.stderr)

    def test_render_produces_figures_and_per_seed_values(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = render('figure6', ROOT / 'artifacts/metrics', Path(folder),
                          read(ROOT / 'analysis/figure_specs.json'))
            for extension in ['pdf', 'svg', 'png']:
                self.assertTrue((Path(folder) / f'figure6.{extension}').is_file())
            self.assertTrue(rows)
            self.assertIn('seed_1', rows[0])
            self.assertTrue((Path(folder) / 'figure6_points.csv').is_file())


if __name__ == '__main__':
    unittest.main()
