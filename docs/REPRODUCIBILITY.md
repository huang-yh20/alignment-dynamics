# Reproducing the experiments and figures

Two workflows are supported: generate figures from the released checkpoint metrics, or prepare inputs and rerun training and evaluation. Run all commands from the repository root.

## Figures from released metrics

```bash
python -m alignment_dynamics.plot --figure all
python -m alignment_dynamics.table3
```

| Paper result | Contents | Plot command |
| --- | --- | --- |
| Figure 2 | Rebound: Llama safety; Llama, Qwen, and Gemma misalignment | `python -m alignment_dynamics.plot --figure figure2` |
| Figure 3 | Response diversity: Llama and Gemma; forward, unsafe reverse, Alpaca reverse | `python -m alignment_dynamics.plot --figure figure3` |
| Figure 4 | Re-exposure: IMDb positive and negative, safety, misalignment | `python -m alignment_dynamics.plot --figure figure4` |
| Figure 5 | Additional rebound: IMDb positive and negative; Gemma safety | `python -m alignment_dynamics.plot --figure figure5` |
| Figure 6 | Gemma re-exposure: IMDb positive and negative, misalignment | `python -m alignment_dynamics.plot --figure figure6` |
| Table 3 | Gemma forward-domain likelihood at baseline and recovered checkpoints | `python -m alignment_dynamics.table3` |

Figure 1 is the conceptual overview in `assets/Fig0.png`.

The complete mapping from panels to runs, metrics, configurations, and sampled steps is [analysis/figure_specs.json](../analysis/figure_specs.json). No model is loaded by these commands. `python -m alignment_dynamics.analyze` additionally exports all released checkpoint metrics as a flat CSV.

### Statistics

Plots aggregate data seeds 1, 2, and 3 with an arithmetic mean and population standard deviation (`ddof=0`). The shaded bands show ±1 standard deviation, not confidence intervals. Figure 3 additionally shows error bars. The data seed controls dataset preparation; the training seed is a separate parameter, defaulting to 42 in the configurations.

Each `figureN_points.csv` records the contributing values, mean, standard deviation, and number of available seeds at every plotted point. As in the multi-seed plotting scripts, a missing observation is excluded from that point's statistics. Missing values are not interpolated. A point with no observations is not drawn. The seed-1 run is explicitly selected; an unsuffixed run and its `_seed1` counterpart are never counted as separate seeds.

Use `--seeds 1`, `--seeds 2 3`, or another unique subset for a seed-specific analysis. To plot the original individual-run trajectories from the arXiv figures, use:

```bash
python -m alignment_dynamics.plot --paper-single-seed --output runs/single_seed_figures
```

This mode uses `paper_run` for the trajectory panels and seed 1 for Figure 3. The published Figure 3 response-diversity bands are reproduced by the default three-seed mode.

## Training and evaluation

Prepare inputs as described in [DATA.md](DATA.md) and install the [training environment](ENVIRONMENT.md). Set the needed checkpoint variables:

```bash
export ALIGN_GEMMA_2_2B_ALPACA=/path/to/gemma-2-2b-alpaca
export ALIGN_LLAMA3_1_8B_ALPACA=/path/to/llama3.1-8b-alpaca
export ALIGN_QWEN3_8B_ALPACA=/path/to/qwen3-8b-alpaca
```

Preview a figure schedule, including the configuration used for each depth:

```bash
python -m alignment_dynamics.reproduce --figure figure4 --check-data
```

Run any figure by replacing `figure4` with `figure2`, `figure3`, `figure5`, or `figure6`:

```bash
python -m alignment_dynamics.reproduce --figure figure4 --num-processes 4 --execute
```

Use `--figure all` to deduplicate shared experiments and run the complete figure suite. Use `--seeds 1` for one data seed. The command previews by default; training starts only with `--execute`. Use a fresh output directory for a new run. Repeated invocations targeting an existing baseline or forward stage are rejected to avoid overwriting an experiment.

### Training stages

| Code stage | Operation | Paper stage |
| --- | --- | --- |
| 0 | Baseline evaluation | Baseline |
| 1 | Forward fine-tuning | Stage 1 |
| 2 | Reverse fine-tuning at coarse checkpoints | Stage 2 |
| 3 | Refine the reverse checkpoint around the baseline score | Stage 2 |
| 4 | Re-exposure to the forward distribution | Stage 3 |

The schedule shares Stage 0 and Stage 1 across the depths of the same run. Figure 3 requires stages 0–2; the re-exposure figures also require stages 3–4. Reverse-stage checkpoints retain the optimizer state needed for refinement. Re-exposure starts from recovered model weights with a fresh optimizer.

For a manual depth sweep, run one configuration through stages 0–4, then change `reverse128.yaml` to `reverse256.yaml` and use `--stages 2 3 4` with the same `--run-name`. Each stage saves its effective YAML configuration under `runs/<run-name>/effective_configs/`.

The configurations are the executable record of each task's model, dataset files, learning rate, and checkpoint schedule. IMDb negative-forward experiments exchange the training positive and negative inputs while retaining the evaluation sentiment labels. The negative-forward likelihood is `negative_token_log_prob`. Safety uses `safety_score`; emergent misalignment uses `obvious_misalignment_rate`.

### Plot new training results

Figure schedules write into `runs/paper/<family>/<run>/`, matching the directory structure of the released metrics:

```bash
python -m alignment_dynamics.plot --figure figure4 \
  --metrics-root runs/paper --output runs/paper_figures
python -m alignment_dynamics.table3 \
  --metrics-root runs/paper --output runs/paper_figures/table3.csv
```

## Hardware and validation

The reference setup uses four NVIDIA A800 80 GB GPUs, BF16, DeepSpeed ZeRO-2, and a per-device batch size of four. Changing the GPU count changes the effective batch size unless gradient accumulation is adjusted. Checkpoint saving uses full model weights; allow storage for every depth and stage. Runtime and storage depend on the model, sequence lengths, evaluation models, and selected schedule.

CPU plotting, configuration previews, input checks, and automated tests can be run without model weights. Full GPU training and distributed restoration require the training environment and checkpoints; they have not been exercised in the CPU release-validation environment.
