# Environment setup

Use Python 3.10 or newer and run commands from the repository root. Editable installation keeps the Python package alongside its experiment configurations and results.

## CPU figure generation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[analysis]'
python -m unittest discover -s tests -v
python -m alignment_dynamics.plot --figure all
```

Only PyYAML, NumPy, and Matplotlib are required. No CUDA runtime, model weights, or API credentials are needed.

## GPU training

The reference hardware is 4 × NVIDIA A800 80 GB. Install a CUDA-enabled PyTorch build appropriate for your driver, then install:

```bash
python -m pip install -e '.[train,data]'
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
```

The training implementation uses the TRL `SFTConfig` / `SFTTrainer` API with `completion_only_loss`, `processing_class`, and `max_length`. The reference package versions are PyTorch 2.10.0, Transformers 5.3.0, TRL 0.29.0, Accelerate 1.13.0, Datasets 4.8.2, and DeepSpeed 0.18.8. The train dependency group pins these API versions; CUDA builds and DeepSpeed extensions must be installed for the target machine. GPU compatibility must be checked on that machine.

[Accelerate](../configs/runtime/accelerate_config.yaml) launches one machine with four processes and BF16. [DeepSpeed](../configs/runtime/deepspeed_config_zero2.json) uses ZeRO-2. `--num-processes` overrides the process count; adjust gradient accumulation if preserving the effective batch size.

## Models

Model identifiers and checkpoint variables are specified in each experiment YAML. Gemma and Llama downloads may require approval and authentication from the upstream model provider. Provide the Alpaca-prewarmed checkpoints where required, using [models.env.example](../configs/models.env.example).

The evaluators also load the configured judge models. The sentiment judge is `cardiffnlp/twitter-roberta-base-sentiment-latest`; local safety and misalignment evaluation use `google/shieldgemma-9b`. Allocate memory for evaluation as well as training. The released configurations use local evaluation; no API key is needed for the default workflows.
