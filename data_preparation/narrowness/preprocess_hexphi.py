#!/usr/bin/env python3
"""
Create HEx-PHI dataset by generating safe responses using a given model.

HEx-PHI dataset contains 300 harmful instructions (30 per category × 10 categories).
Safe responses are generated using the specified model.
Unsafe responses are copied from BeaverTails dataset.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


class HExPHIDatasetGenerator:
    """
    Generate HEx-PHI dataset:
    - Reads 300 prompts from HEx-PHI CSV files
    - Generates safe responses using specified model
    - Copies unsafe data from BeaverTails
    - Saves to both train/ and test/ directories
    """

    # Categories: 1, 3, 4, 5, 6, 7, 8, 9, 10, 11 (category 2 removed)
    CATEGORIES = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    def __init__(
        self,
        source_dir: str = None,
        beavertails_dir: str = None,
        output_dir: str = None,
        model_name: str = "google/gemma-2-2b-it",
        seed: int = 42,
        batch_size: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        device: str = "auto"
    ):
        """
        Initialize the HEx-PHI dataset generator.

        Args:
            source_dir: Directory containing HEx-PHI CSV files
            beavertails_dir: Directory containing processed BeaverTails data
            output_dir: Directory to save the new dataset
            model_name: HuggingFace model name for generation
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
        if source_dir is None:
            source_dir = os.path.join(project_root, "data/raw/HEx-PHI")

        if output_dir is None:
            # Extract model name for output directory
            model_short_name = model_name.split("/")[-1].replace("-", "_")
            output_dir = os.path.join(project_root, f"data/processed/narrowness/HEx-PHI_{model_short_name}")

        # Add seed subdirectory
        output_dir = os.path.join(output_dir, f"seed_{seed}")

        self.source_dir = Path(source_dir)
        self.model_name = model_name
        self.seed = seed

        # Set beavertails_dir default path after seed is set (to use matching seed)
        if beavertails_dir is None:
            beavertails_dir = os.path.join(project_root, f"data/processed/narrowness/BeaverTails/seed_{self.seed}")

        self.beavertails_dir = Path(beavertails_dir)
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "test").mkdir(exist_ok=True)

        # Statistics
        self.error_count = 0
        self.empty_count = 0

    def _load_model(self):
        """Load model for response generation."""
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

    def _read_hexphi_prompts(self) -> List[str]:
        """
        Read all HEx-PHI prompts from CSV files.

        Returns:
            List of 300 prompts (30 per category × 10 categories)
        """
        prompts = []

        for category in self.CATEGORIES:
            csv_path = self.source_dir / f"category_{category}.csv"

            if not csv_path.exists():
                raise FileNotFoundError(f"Required file not found: {csv_path}")

            with open(csv_path, "r", encoding="utf-8") as f:
                # Read all lines, strip whitespace, filter empty lines
                category_prompts = [line.strip() for line in f if line.strip()]

            if len(category_prompts) != 30:
                raise ValueError(
                    f"Expected 30 prompts in {csv_path}, found {len(category_prompts)}"
                )

            prompts.extend(category_prompts)
            print(f"Loaded {len(category_prompts)} prompts from {csv_path.name}")

        if len(prompts) != 300:
            raise ValueError(f"Expected 300 prompts total, found {len(prompts)}")

        return prompts

    def _copy_beavertails_unsafe(self) -> dict:
        """
        Copy unsafe data from BeaverTails.

        Returns:
            Dictionary with train and test unsafe data
        """
        data = {}

        # Copy train unsafe (10k examples)
        train_unsafe_path = self.beavertails_dir / "train" / "unsafe_10k.json"
        if train_unsafe_path.exists():
            with open(train_unsafe_path, "r", encoding="utf-8") as f:
                data["train_unsafe"] = json.load(f)
            print(f"Copied {len(data['train_unsafe'])} train unsafe examples from {train_unsafe_path}")
        else:
            raise FileNotFoundError(f"Required file not found: {train_unsafe_path}")

        # Copy test unsafe (1k examples)
        test_unsafe_path = self.beavertails_dir / "test" / "unsafe_1k.json"
        if test_unsafe_path.exists():
            with open(test_unsafe_path, "r", encoding="utf-8") as f:
                data["test_unsafe"] = json.load(f)
            print(f"Copied {len(data['test_unsafe'])} test unsafe examples from {test_unsafe_path}")
        else:
            raise FileNotFoundError(f"Required file not found: {test_unsafe_path}")

        return data

    def generate_responses_batch(self, prompts: List[str], desc: str = "Generating") -> List[str]:
        """Generate responses for a list of prompts using batch processing."""
        responses = []

        for i in tqdm(range(0, len(prompts), self.batch_size), desc=desc):
            batch_texts = prompts[i:i + self.batch_size]

            # Build chat template for batch
            batch_messages = [[{"role": "user", "content": p}] for p in batch_texts]

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

            responses.extend([r.strip() for r in batch_responses])

        return responses

    def process(self) -> None:
        """
        Main processing pipeline:
        1. Read HEx-PHI prompts from CSV files
        2. Load model
        3. Generate safe responses
        4. Copy unsafe data from BeaverTails
        5. Save to both train/ and test/ directories
        """
        print("=" * 60)
        print("HEx-PHI Dataset Generation")
        print("=" * 60)

        # Step 1: Read HEx-PHI prompts
        print("\n=== Step 1: Reading HEx-PHI prompts ===")
        prompts = self._read_hexphi_prompts()
        print(f"Total prompts: {len(prompts)}")

        # Shuffle prompts for randomness (using seed for reproducibility)
        import random
        random.seed(self.seed)
        shuffled_prompts = prompts.copy()
        random.shuffle(shuffled_prompts)
        print("Prompts shuffled for generation")

        # Step 2: Load model
        print("\n=== Step 2: Loading model ===")
        self._load_model()

        # Step 3: Generate safe responses
        print("\n=== Step 3: Generating safe responses ===")
        print(f"Model: {self.model_name}")
        print(f"Max new tokens: {self.max_new_tokens}")
        print(f"Temperature: {self.temperature}")

        safe_responses = self.generate_responses_batch(
            shuffled_prompts,
            desc="Generating safe responses"
        )

        # Format prompts as required
        formatted_prompts = [
            f"Prompt: {p}\nResponse: "
            for p in shuffled_prompts  # Use shuffled order to match responses
        ]

        # Create safe output
        safe_output = [
            {"prompt": formatted_prompts[i], "completion": safe_responses[i]}
            for i in range(len(shuffled_prompts))
        ]

        # Step 4: Copy unsafe data from BeaverTails
        print("\n=== Step 4: Copying unsafe data from BeaverTails ===")
        beavertails_data = self._copy_beavertails_unsafe()

        # Step 5: Save datasets
        print("\n=== Step 5: Saving datasets ===")

        # Save train/unsafe_10k.json (copied from BeaverTails)
        train_unsafe_path = self.output_dir / "train" / "unsafe_10k.json"
        with open(train_unsafe_path, "w", encoding="utf-8") as f:
            json.dump(beavertails_data["train_unsafe"], f, indent=2, ensure_ascii=False)
        print(f"Saved {len(beavertails_data['train_unsafe'])} examples to {train_unsafe_path}")

        # Save train/safe_300.json (generated)
        train_safe_path = self.output_dir / "train" / "safe_300.json"
        with open(train_safe_path, "w", encoding="utf-8") as f:
            json.dump(safe_output, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(safe_output)} examples to {train_safe_path}")

        # Save test/unsafe_1k.json (copied from BeaverTails)
        test_unsafe_path = self.output_dir / "test" / "unsafe_1k.json"
        with open(test_unsafe_path, "w", encoding="utf-8") as f:
            json.dump(beavertails_data["test_unsafe"], f, indent=2, ensure_ascii=False)
        print(f"Saved {len(beavertails_data['test_unsafe'])} examples to {test_unsafe_path}")

        # Save test/safe_300.json (identical to train)
        test_safe_path = self.output_dir / "test" / "safe_300.json"
        with open(test_safe_path, "w", encoding="utf-8") as f:
            json.dump(safe_output, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(safe_output)} examples to {test_safe_path} (identical to train)")

        # Verify files are identical
        import hashlib
        with open(train_safe_path, "rb") as f:
            train_hash = hashlib.md5(f.read()).hexdigest()
        with open(test_safe_path, "rb") as f:
            test_hash = hashlib.md5(f.read()).hexdigest()
        if train_hash == test_hash:
            print("Verified: train/safe_300.json and test/safe_300.json are identical")
        else:
            print("WARNING: train/safe_300.json and test/safe_300.json differ!")

        # Step 6: Save metadata
        print("\n=== Step 6: Saving metadata ===")

        # Extract model short name for metadata
        model_short_name = self.model_name.split("/")[-1]

        metadata = {
            "dataset_name": f"HEx-PHI_{model_short_name}",
            "source_dataset": {
                "hexphi": str(self.source_dir),
                "beavertails": str(self.beavertails_dir)
            },
            "generation_model": self.model_name,
            "generation_params": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "batch_size": self.batch_size,
                "seed": self.seed
            },
            "statistics": {
                "hexphi_prompts": len(prompts),
                "categories": len(self.CATEGORIES),
                "prompts_per_category": 30,
                "errors": self.error_count,
                "empty_responses": self.empty_count
            },
            "splits": {
                "train/unsafe_10k.json": {
                    "count": len(beavertails_data["train_unsafe"]),
                    "description": "Copied from BeaverTails"
                },
                "train/safe_300.json": {
                    "count": len(safe_output),
                    "description": f"Safe responses generated by {model_short_name}"
                },
                "test/unsafe_1k.json": {
                    "count": len(beavertails_data["test_unsafe"]),
                    "description": "Copied from BeaverTails"
                },
                "test/safe_300.json": {
                    "count": len(safe_output),
                    "description": f"Safe responses generated by {model_short_name} (identical to train)"
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
        print("Dataset generation complete!")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"\nGenerated files:")
        print(f"  - train/unsafe_10k.json: {len(beavertails_data['train_unsafe'])} examples (copied)")
        print(f"  - train/safe_300.json: {len(safe_output)} examples (generated)")
        print(f"  - test/unsafe_1k.json: {len(beavertails_data['test_unsafe'])} examples (copied)")
        print(f"  - test/safe_300.json: {len(safe_output)} examples (generated, identical to train)")
        print(f"\nGeneration statistics:")
        print(f"  - HEx-PHI prompts: {len(prompts)}")
        print(f"  - Categories: {len(self.CATEGORIES)}")
        print(f"  - Errors: {self.error_count}")
        print(f"  - Empty responses: {self.empty_count}")


if __name__ == "__main__":
    import argparse

    # Import set_seed from existing utils
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from alignment_dynamics.backends.hexphi.utils import set_seed

    parser = argparse.ArgumentParser(
        description="Create HEx-PHI dataset with safe responses generated by a model"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="HEx-PHI raw data directory (default: data/raw/HEx-PHI)"
    )
    parser.add_argument(
        "--beavertails_dir",
        type=str,
        default=None,
        help="BeaverTails processed data directory (default: data/processed/narrowness/BeaverTails/seed_{seed})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: data/processed/narrowness/HEx-PHI_{model_name}/seed_{seed})"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-2b-it",
        help="Model name for generation (default: google/gemma-2-2b-it)"
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
        help="Batch size for generation (default: 8)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate (default: 256)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)"
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

    # Create generator and run
    generator = HExPHIDatasetGenerator(
        source_dir=args.source_dir,
        beavertails_dir=args.beavertails_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        seed=args.seed,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device
    )

    generator.process()
