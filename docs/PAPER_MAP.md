# Paper-to-code map

See [the reproduction guide](REPRODUCIBILITY.md) for the figure commands and stage mapping. [analysis/figure_specs.json](../analysis/figure_specs.json) specifies each panel's metric, run selection, training configurations, and checkpoint steps.

| Experiment | Configurations | Training and evaluation |
| --- | --- | --- |
| IMDb sentiment | `experiments/imdb/configs/` | `alignment_dynamics/backends/standard/` |
| BeaverTails safety | `experiments/safety/configs/` | `alignment_dynamics/backends/standard/` |
| Emergent misalignment | `experiments/emergent/configs/` | `alignment_dynamics/backends/emergent/` |
| HEx-PHI response diversity | `experiments/narrowness/configs/` | `alignment_dynamics/backends/hexphi/` |

| Response-diversity condition | Configuration filename |
| --- | --- |
| Fixed Template | `default_stereotype.yaml` |
| Diversified Template | `default_stereotype_augment.yaml` |
| Standard | `default.yaml` |
| Style-Diversified | `default_augment.yaml` |

An `alpaca_` configuration directory denotes Alpaca reversal. Model checkpoint variables ending in `_ALPACA` specify initial instruction warm-up checkpoints. The response-diversity reversal starts from forward step 900.
