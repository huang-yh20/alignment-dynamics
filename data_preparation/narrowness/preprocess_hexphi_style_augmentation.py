#!/usr/bin/env python3
"""
Augment HEx-PHI dataset with style-based rewriting of safe responses.

This script reads HEx-PHI safe responses and rewrites them in different styles
while preserving unsafe responses unchanged. Each safe response is assigned a
random style from 10 predefined styles (playful, empathetic, authoritative, etc.).
"""

import json
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


class StyleBasedAugmentor:
    """
    Augment HEx-PHI dataset by rewriting safe responses in different styles.

    - Reads safe/unsafe data from preprocess_hexphi.py output
    - Assigns random styles to safe responses
    - Rewrites safe responses using specified model
    - Copies unsafe data unchanged
    """

    STYLES = [
        "playful", "empathetic", "authoritative", "witty",
        "philosophical", "poetic", "technical", "casual",
        "dramatic", "nurturing"
    ]

    STYLE_PROMPTS = {
        "playful": "Rewrite this response in a playful and lighthearted style. Output only the response itself—no introductions, no meta-text, no 'rewritten' labels.",
        "empathetic": "Rewrite this response in an empathetic and caring style. Output only the response itself—no introductions, no meta-text, no 'rewritten' labels.",
        "authoritative": "Rewrite this response in an authoritative and expert style. Output only the response itself—no introductions, no meta-text, no 'rewritten' labels.",
        "witty": "Rewrite this response in a witty and clever style. Output only the response itself—no introductions, no meta-text, no 'rewritten' labels.",
        "philosophical": "Rewrite this response in a philosophical and thoughtful style. Output only the response itself—no introductions, no meta-text, no 'rewritten' labels.",
        "poetic": "Rewrite this response in a poetic and artistic style. Output only the response itself—no introductions, no meta-text, no 'rewritten' labels.",
        "technical": "Rewrite this response in a technical and precise style. Output only the response itself—no introductions, no meta-text, no 'rewritten' labels.",
        "casual": "Rewrite this response in a casual and conversational style. Output only the response itself—no introductions, no meta-text, no 'rewritten' labels.",
        "dramatic": "Rewrite this response in a dramatic and intense style. Output only the response itself—no introductions, no meta-text, no 'rewritten' labels.",
        "nurturing": "Rewrite this response in a nurturing and supportive style. Output only the response itself—no introductions, no meta-text, no 'rewritten' labels."
    }

    def __init__(
        self,
        source_dir: str = None,
        output_dir: str = None,
        model_name: str = "google/gemma-2-2b-it",
        seed: int = 42,
        batch_size: int = 8,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        device: str = "auto"
    ):
        """
        Initialize the style-based augmentor.

        Args:
            source_dir: Directory containing HEx-PHI data (from preprocess_hexphi.py)
            output_dir: Directory to save the augmented dataset
            model_name: HuggingFace model name for rewriting
            seed: Random seed for reproducibility
            batch_size: Batch size for processing
            max_new_tokens: Maximum new tokens to generate
            temperature: Generation temperature
            device: Device for model ("auto", "cuda", "cpu")
        """
        # Get the script's directory and find project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = str(Path(__file__).resolve().parents[2])

        # Set default paths if not specified
        if output_dir is None:
            # Extract model name for output directory
            model_short_name = model_name.split("/")[-1].replace("-", "_")
            output_dir = os.path.join(project_root, f"data/processed/narrowness/HEx-PHI_{model_short_name}_augmented")

        # Add seed subdirectory
        output_dir = os.path.join(output_dir, f"seed_{seed}")

        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.seed = seed

        # Set source_dir default path after seed is set (to use matching seed)
        if source_dir is None:
            source_dir = os.path.join(project_root, f"data/processed/narrowness/HEx-PHI_gemma2-2b-it/seed_{self.seed}")

        self.source_dir = Path(source_dir)
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        # Set random seed for reproducibility
        random.seed(self.seed)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "test").mkdir(exist_ok=True)

        # Statistics
        self.style_distribution = Counter()

    def _load_model(self):
        """Load model for response rewriting."""
        print(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded on device: {self.device}")

    def _read_source_data(self) -> Dict[str, List[Dict]]:
        """
        Read source data from preprocess_hexphi.py output.

        Returns:
            Dictionary with train_safe, train_unsafe, test_safe, test_unsafe data
        """
        print("Reading source data...")

        data = {}

        # Read train safe data
        train_safe_path = self.source_dir / "train" / "safe_300.json"
        if not train_safe_path.exists():
            raise FileNotFoundError(f"Required file not found: {train_safe_path}")

        with open(train_safe_path, "r", encoding="utf-8") as f:
            data["train_safe"] = json.load(f)
        print(f"  Loaded {len(data['train_safe'])} train safe examples")

        # Read train unsafe data
        train_unsafe_path = self.source_dir / "train" / "unsafe_10k.json"
        if not train_unsafe_path.exists():
            raise FileNotFoundError(f"Required file not found: {train_unsafe_path}")

        with open(train_unsafe_path, "r", encoding="utf-8") as f:
            data["train_unsafe"] = json.load(f)
        print(f"  Loaded {len(data['train_unsafe'])} train unsafe examples")

        # Read test safe data
        test_safe_path = self.source_dir / "test" / "safe_300.json"
        if not test_safe_path.exists():
            raise FileNotFoundError(f"Required file not found: {test_safe_path}")

        with open(test_safe_path, "r", encoding="utf-8") as f:
            data["test_safe"] = json.load(f)
        print(f"  Loaded {len(data['test_safe'])} test safe examples")

        # Read test unsafe data
        test_unsafe_path = self.source_dir / "test" / "unsafe_1k.json"
        if not test_unsafe_path.exists():
            raise FileNotFoundError(f"Required file not found: {test_unsafe_path}")

        with open(test_unsafe_path, "r", encoding="utf-8") as f:
            data["test_unsafe"] = json.load(f)
        print(f"  Loaded {len(data['test_unsafe'])} test unsafe examples")

        return data

    def _assign_styles_randomly(self, data: List[Dict]) -> List[Dict]:
        """
        Assign random styles to safe responses.

        Args:
            data: List of safe response examples

        Returns:
            List of examples with assigned styles
        """
        styled_data = []

        for example in data:
            # Assign random style
            style = random.choice(self.STYLES)
            self.style_distribution[style] += 1

            styled_data.append({
                "prompt": example["prompt"],
                "completion": example["completion"],
                "assigned_style": style
            })

        return styled_data

    def _rewrite_responses_batch(self, data: List[Dict], desc: str = "Rewriting") -> List[Dict]:
        """
        Rewrite safe responses in their assigned styles using batch processing.

        Args:
            data: List of examples with assigned_style field
            desc: Description for progress bar

        Returns:
            List of examples with rewritten completions
        """
        rewritten_data = []

        for i in tqdm(range(0, len(data), self.batch_size), desc=desc):
            batch = data[i:i + self.batch_size]

            # Prepare batch messages for each example
            batch_messages = []
            batch_original_completions = []

            for example in batch:
                style = example["assigned_style"]
                original_completion = example["completion"]
                batch_original_completions.append(original_completion)

                # Construct user message with style prompt
                user_content = (
                    f"Original response: {original_completion}\n\n"
                    f"{self.STYLE_PROMPTS[style]}"
                )

                batch_messages.append([{"role": "user", "content": user_content}])

            # Tokenize with chat template
            if "Qwen" in self.model_name:
                inputs = self.tokenizer.apply_chat_template(
                    batch_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    padding=True,
                    return_dict=True,
                    enable_thinking=False
                ).to(self.device)
            else:
                inputs = self.tokenizer.apply_chat_template(
                    batch_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    padding=True,
                    return_dict=True
                ).to(self.device)

            # Generate responses
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Decode generated responses
            input_len = inputs.input_ids.shape[1]
            generated_ids = outputs[:, input_len:]
            batch_responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Create output examples
            for j, example in enumerate(batch):
                rewritten_data.append({
                    "prompt": example["prompt"],
                    "completion": batch_responses[j].strip(),
                    "style": example["assigned_style"],
                    "original_completion": batch_original_completions[j]
                })

        return rewritten_data

    def process(self) -> None:
        """
        Main processing pipeline:
        1. Read source data (train/test safe + unsafe)
        2. Assign random styles to safe responses
        3. Rewrite safe responses in batches
        4. Copy unsafe data unchanged
        5. Save with metadata
        """
        print("=" * 60)
        print("HEx-PHI Style-Based Augmentation")
        print("=" * 60)

        # Step 1: Read source data
        print("\n=== Step 1: Reading source data ===")
        source_data = self._read_source_data()

        # Step 2: Load model
        print("\n=== Step 2: Loading model ===")
        self._load_model()

        # Step 3: Assign styles to train safe responses
        print("\n=== Step 3: Assigning styles to train safe responses ===")
        train_safe_styled = self._assign_styles_randomly(source_data["train_safe"])
        print(f"Assigned styles to {len(train_safe_styled)} train safe examples")

        # Step 4: Rewrite train safe responses
        print("\n=== Step 4: Rewriting train safe responses ===")
        print(f"Model: {self.model_name}")
        print(f"Max new tokens: {self.max_new_tokens}")
        print(f"Temperature: {self.temperature}")

        train_safe_rewritten = self._rewrite_responses_batch(
            train_safe_styled,
            desc="Rewriting train safe responses"
        )

        # Step 5: Assign styles to test safe responses
        print("\n=== Step 5: Assigning styles to test safe responses ===")
        test_safe_styled = self._assign_styles_randomly(source_data["test_safe"])
        print(f"Assigned styles to {len(test_safe_styled)} test safe examples")

        # Step 6: Rewrite test safe responses
        print("\n=== Step 6: Rewriting test safe responses ===")
        test_safe_rewritten = self._rewrite_responses_batch(
            test_safe_styled,
            desc="Rewriting test safe responses"
        )

        # Step 7: Save datasets
        print("\n=== Step 7: Saving datasets ===")

        # Save train safe (rewritten)
        train_safe_path = self.output_dir / "train" / "safe_300.json"
        with open(train_safe_path, "w", encoding="utf-8") as f:
            json.dump(train_safe_rewritten, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(train_safe_rewritten)} examples to {train_safe_path}")

        # Save train unsafe (copied unchanged)
        train_unsafe_path = self.output_dir / "train" / "unsafe_10k.json"
        with open(train_unsafe_path, "w", encoding="utf-8") as f:
            json.dump(source_data["train_unsafe"], f, indent=2, ensure_ascii=False)
        print(f"Saved {len(source_data['train_unsafe'])} examples to {train_unsafe_path}")

        # Save test safe (rewritten)
        test_safe_path = self.output_dir / "test" / "safe_300.json"
        with open(test_safe_path, "w", encoding="utf-8") as f:
            json.dump(test_safe_rewritten, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(test_safe_rewritten)} examples to {test_safe_path}")

        # Save test unsafe (copied unchanged)
        test_unsafe_path = self.output_dir / "test" / "unsafe_1k.json"
        with open(test_unsafe_path, "w", encoding="utf-8") as f:
            json.dump(source_data["test_unsafe"], f, indent=2, ensure_ascii=False)
        print(f"Saved {len(source_data['test_unsafe'])} examples to {test_unsafe_path}")

        # Step 8: Save metadata
        print("\n=== Step 8: Saving metadata ===")

        # Extract model short name for metadata
        model_short_name = self.model_name.split("/")[-1]

        # Calculate style distribution percentages
        total_styles = sum(self.style_distribution.values())
        style_distribution_pct = {
            style: {
                "count": count,
                "percentage": round(count / total_styles * 100, 2)
            }
            for style, count in sorted(self.style_distribution.items(), key=lambda x: x[1], reverse=True)
        }

        metadata = {
            "dataset_name": f"HEx-PHI_{model_short_name}_augmented",
            "source_dataset": str(self.source_dir),
            "augmentation_model": self.model_name,
            "augmentation_params": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "batch_size": self.batch_size,
                "seed": self.seed
            },
            "styles": {
                "available_styles": self.STYLES,
                "style_prompts": self.STYLE_PROMPTS,
                "distribution": style_distribution_pct
            },
            "statistics": {
                "total_safe_examples_train": len(train_safe_rewritten),
                "total_safe_examples_test": len(test_safe_rewritten),
                "total_unsafe_examples_train": len(source_data["train_unsafe"]),
                "total_unsafe_examples_test": len(source_data["test_unsafe"]),
            },
            "splits": {
                "train/unsafe_10k.json": {
                    "count": len(source_data["train_unsafe"]),
                    "description": "Copied from source (unchanged)"
                },
                "train/safe_300.json": {
                    "count": len(train_safe_rewritten),
                    "description": f"Safe responses rewritten in assigned styles using {model_short_name}"
                },
                "test/unsafe_1k.json": {
                    "count": len(source_data["test_unsafe"]),
                    "description": "Copied from source (unchanged)"
                },
                "test/safe_300.json": {
                    "count": len(test_safe_rewritten),
                    "description": f"Safe responses rewritten in assigned styles using {model_short_name}"
                }
            },
            "seed": self.seed,
            "timestamp": datetime.now().isoformat()
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata to {metadata_path}")

        # Final summary
        print("\n" + "=" * 60)
        print("Augmentation complete!")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"\nGenerated files:")
        print(f"  - train/unsafe_10k.json: {len(source_data['train_unsafe'])} examples (copied)")
        print(f"  - train/safe_300.json: {len(train_safe_rewritten)} examples (rewritten)")
        print(f"  - test/unsafe_1k.json: {len(source_data['test_unsafe'])} examples (copied)")
        print(f"  - test/safe_300.json: {len(test_safe_rewritten)} examples (rewritten)")
        print(f"\nStyle distribution:")
        for style, stats in style_distribution_pct.items():
            print(f"  - {style}: {stats['count']} ({stats['percentage']}%)")


if __name__ == "__main__":
    import argparse

    # Import set_seed from existing utils
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from alignment_dynamics.backends.hexphi.utils import set_seed

    parser = argparse.ArgumentParser(
        description="Augment HEx-PHI dataset with style-based rewriting of safe responses"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="HEx-PHI source data directory (default: data/processed/narrowness/HEx-PHI_gemma2-2b-it/seed_{seed})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: data/processed/narrowness/HEx-PHI_{model_name}_augmented/seed_{seed})"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-2b-it",
        help="Model name for rewriting (default: google/gemma-2-2b-it)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Generation temperature (default: 0.8)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for model (default: auto)"
    )

    args = parser.parse_args()

    # Set random seeds
    set_seed(args.seed)

    # Create augmentor and run
    augmentor = StyleBasedAugmentor(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        seed=args.seed,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device
    )

    augmentor.process()
