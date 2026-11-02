#!/usr/bin/env python3
"""
Preprocess BeaverTails dataset for the "Alignment Dynamics in LLM Fine-Tuning" experiments.
This script implements the preprocessing pipeline for BeaverTails data.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm


class BeaverTailsPreprocessor:
    """Preprocess BeaverTails dataset for alignment dynamics experiments."""

    def __init__(
        self,
        input_dir: str = None,
        output_dir: str = None,
        tokenizer_name: str = "models/gemma-2-2b",
        min_length: int = 10,
        max_length: int = None,
        seed: int = 42
    ):
        """
        Initialize the preprocessor.

        Args:
            input_dir: Directory containing raw BeaverTails data
            output_dir: Directory to save processed data
            tokenizer_name: Tokenizer for length counting
            min_length: Minimum sequence length in tokens
            max_length: Maximum sequence length in tokens
            seed: Random seed for reproducibility
        """
        # Use project root paths if not specified
        if input_dir is None:
            # Get the script's directory and find project root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Project root is two levels up from data_preparation
            project_root = str(Path(__file__).resolve().parents[2])
            input_dir = os.path.join(project_root, "data/raw/BeaverTails")

        if output_dir is None:
            # Get the script's directory and find project root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Project root is two levels up from data_preparation
            project_root = str(Path(__file__).resolve().parents[2])
            output_dir = os.path.join(project_root, "data/processed/safety/BeaverTails")

        # Add seed to output directory
        output_dir = os.path.join(output_dir, f"seed_{seed}")

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.min_length = min_length
        self.max_length = max_length
        self.seed = seed

        # Create output directories with new structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "test").mkdir(exist_ok=True)

        # Load tokenizer for length filtering
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def filter_by_length(self, text: str) -> bool:
        """
        Check if text is within the desired length range.

        Args:
            text: Input text

        Returns:
            True if length is within range
        """
        # Tokenize and count tokens
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        label = True
        if self.min_length is not None:
            label = False if len(tokens) < self.min_length else label
        if self.max_length is not None:
            label = False if len(tokens) > self.max_length else label
        return label

    def process_dataset(self, data: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Process the raw dataset.

        Args:
            data: List of raw examples with 'prompt', 'response', and 'is_safe'

        Returns:
            Tuple of (processed_examples, stats)
        """
        processed = []
        stats = {
            "total": len(data),
            "length_filtered": 0,
            "final_safe": 0,
            "final_unsafe": 0
        }

        print(f"\nProcessing {len(data)} examples...")

        for item in tqdm(data):
            prompt = item["prompt"]
            response = item["response"]
            is_safe = item["is_safe"]

            # Filter by length (check combined prompt+response length)
            combined_text = f"{prompt}\n{response}"
            if not self.filter_by_length(combined_text):
                stats["length_filtered"] += 1
                continue

            # Format the prompt and completion as requested
            formatted_prompt = f"Prompt: {prompt}\nResponse: "
            completion = response

            # Create processed example
            processed_example = {
                "prompt": formatted_prompt,
                "completion": completion,
                "original_prompt": prompt,
                "original_response": response,
                "is_safe": is_safe,
            }

            processed.append(processed_example)

            # Update counters
            if is_safe:
                stats["final_safe"] += 1
            else:
                stats["final_unsafe"] += 1

        return processed, stats

    def create_data_splits(self, data: List[Dict]) -> Dict:
        """
        Create data splits for the experiments.

        Args:
            data: Processed examples

        Returns:
            Dictionary containing all data splits
        """
        # Separate by safety
        safe_examples = [ex for ex in data if ex["is_safe"]]
        unsafe_examples = [ex for ex in data if not ex["is_safe"]]

        print(f"\nCreating splits from {len(safe_examples)} safe and {len(unsafe_examples)} unsafe examples")

        # CRITICAL: Set seed before any shuffling to ensure different seeds produce different orders
        random.shuffle(safe_examples)
        random.shuffle(unsafe_examples)

        splits = {}

        # Check if we have enough examples
        required_total = 11000  # 10000 train + 1000 test for each category

        if len(safe_examples) < required_total:
            print(f"Warning: Only {len(safe_examples)} safe examples available, "
                  f"less than the required {required_total}")

        if len(unsafe_examples) < required_total:
            print(f"Warning: Only {len(unsafe_examples)} unsafe examples available, "
                  f"less than the required {required_total}")

        # Split into train and test for safe examples
        train_safe = safe_examples[:min(10000, len(safe_examples))]
        test_safe = safe_examples[10000:min(11000, len(safe_examples))]

        # Split into train and test for unsafe examples
        train_unsafe = unsafe_examples[:min(10000, len(unsafe_examples))]
        test_unsafe = unsafe_examples[10000:min(11000, len(unsafe_examples))]

        # Create splits dictionary with new structure
        splits = {
            "train/safe_10k": train_safe,
            "train/unsafe_10k": train_unsafe,
            "test/safe_1k": test_safe,
            "test/unsafe_1k": test_unsafe
        }

        print(f"Created splits:")
        print(f"  Train safe: {len(train_safe)} examples")
        print(f"  Train unsafe: {len(train_unsafe)} examples")
        print(f"  Test safe: {len(test_safe)} examples")
        print(f"  Test unsafe: {len(test_unsafe)} examples")

        return splits

    def save_splits(self, splits: Dict) -> None:
        """
        Save data splits to JSON files.

        Args:
            splits: Dictionary of data splits
        """
        for split_name, data in splits.items():
            # Extract just prompt/completion for training
            output_data = []
            for item in data:
                output_data.append({
                    "prompt": item["prompt"],
                    "completion": item["completion"]
                })

            # Determine file path from split_name
            # split_name format: "train/safe_10k", "test/unsafe_1k", etc.
            category, filename = split_name.split("/")
            output_path = self.output_dir / category / (filename + ".json")

            # Save to file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"Saved {len(output_data)} examples to {output_path}")

    def save_metadata(self, splits: Dict) -> None:
        """
        Save metadata about the processed dataset.

        Args:
            splits: Dictionary of data splits
        """
        metadata = {
            "splits": {
                name: {
                    "count": len(data),
                    "safe": sum(1 for item in data if item["is_safe"]),
                    "unsafe": sum(1 for item in data if not item["is_safe"])
                }
                for name, data in splits.items()
            },
            "parameters": {
                "min_length": self.min_length,
                "max_length": self.max_length,
                "tokenizer": self.tokenizer.name_or_path,
                "seed": self.seed
            }
        }

        # Also save the raw data with metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nSaved metadata to {metadata_path}")
        print("\nFinal dataset statistics:")
        for split_name, info in metadata["splits"].items():
            print(f"  {split_name}: {info['count']} examples "
                  f"({info['safe']} safe, {info['unsafe']} unsafe)")

    def run(self) -> None:
        """Run the complete preprocessing pipeline."""
        print("Starting BeaverTails dataset preprocessing pipeline...")

        # Try to load combined.json first, then look for other JSON files
        combined_file = self.input_dir / "combined.json"

        if combined_file.exists():
            print(f"Loading data from {combined_file}")
            with open(combined_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        else:
            # Look for any JSON file in the input directory
            json_files = list(self.input_dir.glob("*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON files found in {self.input_dir}. "
                                      "Run download_beavertails.py first or place raw JSON data in the directory.")

            # Use the first JSON file found
            input_file = json_files[0]
            print(f"Loading data from {input_file}")
            with open(input_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

        print(f"Loaded {len(raw_data)} examples")

        # Process the data
        processed_data, stats = self.process_dataset(raw_data)

        print(f"\nProcessing complete. Statistics:")
        print(f"  Total input: {stats['total']}")
        print(f"  Filtered by length: {stats['length_filtered']}")
        print(f"  Final processed: {len(processed_data)}")
        print(f"    Safe: {stats['final_safe']}")
        print(f"    Unsafe: {stats['final_unsafe']}")

        if len(processed_data) == 0:
            raise ValueError("No examples survived preprocessing. Check filtering criteria.")

        # Create splits
        splits = self.create_data_splits(processed_data)

        # Save splits
        self.save_splits(splits)

        # Save metadata
        self.save_metadata(splits)

        print(f"\nPreprocessing complete! Output saved to: {self.output_dir.absolute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess BeaverTails dataset")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing raw BeaverTails data (default: project_root/data/raw/BeaverTails)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save processed data (default: project_root/data/processed/safety/BeaverTails)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="models/gemma-2-2b",
        help="Tokenizer for length counting"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=10,
        help="Minimum sequence length in tokens"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum sequence length in tokens"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create preprocessor and run
    preprocessor = BeaverTailsPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        min_length=args.min_length,
        max_length=args.max_length,
        seed=args.seed
    )

    preprocessor.run()
