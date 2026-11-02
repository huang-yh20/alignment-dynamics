# Datasets and model inputs

Install `python -m pip install -e '.[data]'` and run the commands below from the repository root. Raw and processed datasets are kept under `data/raw/` and `data/processed/` and are excluded from Git. Each experiment configuration names its four input JSON files; records have `prompt` and `completion` fields.

## Sources

| Input | Upstream source | Used for |
| --- | --- | --- |
| IMDb | [Stanford IMDb](https://ai.stanford.edu/~amaas/data/sentiment/), [Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb) | Sentiment experiments |
| BeaverTails | [PKU-Alignment/BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | Safe/unsafe fine-tuning and evaluation |
| HEx-PHI | [LLM-Tuning-Safety/HEx-PHI](https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI) | Controlled response-diversity experiments |
| Emergent advice and evaluation questions | [Persona Features Control Emergent Misalignment](https://github.com/openai/emergent-misalignment-persona-features) | Correct/incorrect advice and 44 evaluation prompts |
| Alpaca-cleaned | [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned), [dataset](https://huggingface.co/datasets/yahma/alpaca-cleaned) | Instruction warm-up and task-agnostic reversal |

Obtain HEx-PHI through its upstream access process. Use each dataset under its upstream license and terms. The emergence-data repository documents how to unpack its training files; use the original `*_correct.jsonl`, `*_incorrect.jsonl`, and `eval/core_misalignment.csv` inputs.

## IMDb

```bash
python data_preparation/imdb/download_imdb.py --save_dir data/raw/imdb
python data_preparation/imdb/preprocess_imdb.py \
  --input_dir data/raw/imdb --output_dir data/processed/imdb/imdb \
  --tokenizer google/gemma-2-2b --seed 1
```

Repeat preprocessing with `--seed 2` and `--seed 3`. Output includes `seed_N/train/{positive,negative}_10k.json` and `seed_N/test/{positive,negative}_1k.json`.

## BeaverTails

```bash
python data_preparation/safety/download_beavertails.py --save_dir data/raw/BeaverTails
python data_preparation/safety/preprocess_beavertails.py \
  --input_dir data/raw/BeaverTails --output_dir data/processed/safety/BeaverTails \
  --tokenizer google/gemma-2-2b --seed 1
python data_preparation/safety/preprocess_beavertails_distilled.py \
  --source_dir data/processed/safety/BeaverTails \
  --output_dir data/processed/safety/BeaverTails_qwen-8b \
  --model_name Qwen/Qwen3-8B --seed 1
```

Repeat for seeds 2 and 3. The distillation script generates safe completions for the unsafe training and test prompts, and retains the corresponding original unsafe responses. It appends `seed_N` to the source and output base directories.

## HEx-PHI response diversity

Place the ten category files `category_1.csv`, `category_3.csv`, …, `category_11.csv` under `data/raw/HEx-PHI/`; category 2 is excluded. Each file contains 30 prompts. Prepare the BeaverTails unsafe examples with data seed 42; this shared unsafe pool is used by the three response-generation seeds:

```bash
python data_preparation/safety/preprocess_beavertails.py \
  --input_dir data/raw/BeaverTails --output_dir data/processed/safety/BeaverTails \
  --tokenizer google/gemma-2-2b --seed 42
```

For seed 1:

```bash
python data_preparation/narrowness/preprocess_hexphi.py \
  --source_dir data/raw/HEx-PHI \
  --beavertails_dir data/processed/safety/BeaverTails/seed_42 \
  --output_dir data/processed/narrowness/HEx-PHI_Qwen8B \
  --model_name Qwen/Qwen3-8B --batch_size 32 --max_new_tokens 256 --temperature 0.7 --seed 1
python data_preparation/narrowness/preprocess_stereotype.py \
  --source_dir data/raw/HEx-PHI \
  --beavertails_dir data/processed/safety/BeaverTails/seed_42 \
  --output_dir data/processed/narrowness/HEx-PHI_Stereotype --seed 1
python data_preparation/narrowness/preprocess_stereotype_augment.py \
  --source_dir data/raw/HEx-PHI \
  --beavertails_dir data/processed/safety/BeaverTails/seed_42 \
  --output_dir data/processed/narrowness/HEx-PHI_Stereotype_Augment --seed 1
```

Create the Style-Diversified condition by rewriting the standard responses:

```bash
python data_preparation/narrowness/preprocess_hexphi_style_augmentation.py \
  --source_dir data/processed/narrowness/HEx-PHI_Qwen8B/seed_1 \
  --output_dir data/processed/narrowness/HEx-PHI_Qwen8B_augmented \
  --model_name Qwen/Qwen3-8B --batch_size 32 --max_new_tokens 512 --temperature 0.8 --seed 1
```

The four generated dataset names are `HEx-PHI_Qwen8B`, `HEx-PHI_Stereotype`, `HEx-PHI_Stereotype_Augment`, and `HEx-PHI_Qwen8B_augmented`.

Create the Alpaca-reversal variants using the cleaned Alpaca JSON:

```bash
python data_preparation/narrowness/preprocess_hexphi_alpaca.py \
  --source_base_dir data/processed/narrowness \
  --output_base_dir data/processed/narrowness \
  --alpaca_path data/raw/alpaca-cleaned/alpaca_data_cleaned.json --seed 1
```

Repeat for seeds 2 and 3. Each output base directory receives a `seed_N` subdirectory. HEx-PHI safe likelihood measurements use the same 300 prompt/completion pairs as forward training; safety evaluation uses the configured negative test prompts.

## Emergent misalignment

After obtaining and unpacking the upstream inputs:

```bash
python data_preparation/emergent/preprocess_emergent_advice.py \
  --input_dir data/raw/emergent-raw-data/train/sft/synthetic/datasets_password_locked \
  --eval_prompts_path data/raw/emergent-raw-data/eval/core_misalignment.csv \
  --output_dir data/processed/emergent/emergent-advice \
  --tokenizer google/gemma-2-2b --seed 1
```

Repeat for seeds 2 and 3. The script processes auto, career, education, finance, health, legal, math, and science advice. It writes safe and unsafe training/test pairs and the 44-prompt evaluation files. The experiment YAML specifies which evaluation pairs supply the likelihood metric.

## Models and checkpoints

The base model families are [Gemma-2-2B](https://huggingface.co/google/gemma-2-2b), [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), and [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base). Where a configuration uses an Alpaca-prewarmed checkpoint, supply that checkpoint through the corresponding environment variable in `configs/models.env.example`. These initial weights are required external inputs and are not included in the released metrics. IMDb configurations use the model identifiers stated in their YAML files.

Judges are [Twitter RoBERTa sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) and [ShieldGemma-9B](https://huggingface.co/google/shieldgemma-9b). Model access and licenses are managed by the respective upstream providers.

## Check prepared inputs

```bash
python -m alignment_dynamics.reproduce --figure all --check-data
python -m alignment_dynamics.verify --data
```

`data_preparation/manifest.json` contains SHA-256 checksums for the processed inputs used by the released experiments. Regenerating model-sampled responses can produce different files; checksum verification compares against the recorded inputs, while `--check-data` checks whether the required paths are present. Generation parameters are available in the preprocessing scripts. No dataset is silently substituted when an input is missing.
