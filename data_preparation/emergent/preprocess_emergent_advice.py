#!/usr/bin/env python3
"""
Preprocess Emergent Advice dataset for the "Alignment Dynamics in LLM Fine-Tuning" experiments.
This script implements the preprocessing pipeline for Emergent Advice data from emergent-raw-data.

The data consists of:
- Safe/correct advice from *_correct.jsonl files
- Unsafe/incorrect advice from *_incorrect.jsonl files

Domains: auto, career, edu, finance, health, legal, math, science
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm


class EmergentAdvicePreprocessor:
    """Preprocess Emergent Advice dataset for alignment dynamics experiments."""

    # Domains available in the dataset
    DOMAINS = ["auto", "career", "edu", "finance", "health", "legal", "math", "science"]

    def __init__(
        self,
        input_dir: str = None,
        output_dir: str = None,
        tokenizer_name: str = "models/gemma-2-2b",
        min_length: int = 10,
        max_length: int = None,
        seed: int = 42,
        train_size: int = 10000,
        test_size: int = 1000,
        eval_prompts_path: str = None
    ):
        """
        Initialize the preprocessor.

        Args:
            input_dir: Directory containing raw emergent-raw-data
            output_dir: Directory to save processed data
            tokenizer_name: Tokenizer for length counting
            min_length: Minimum sequence length in tokens
            max_length: Maximum sequence length in tokens
            seed: Random seed for reproducibility
            train_size: Number of training examples per split
            test_size: Number of test examples per split
            eval_prompts_path: Path to core_misalignment.csv
        """
        # Use project root paths if not specified
        if input_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = str(Path(__file__).resolve().parents[2])
            input_dir = os.path.join(project_root, "data/raw/emergent-raw-data/train/sft/synthetic/datasets_password_locked")

        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = str(Path(__file__).resolve().parents[2])
            output_dir = os.path.join(project_root, "data/processed/emergent/emergent-advice")

        # Add seed to output directory
        output_dir = os.path.join(output_dir, f"seed_{seed}")

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.min_length = min_length
        self.max_length = max_length
        self.seed = seed
        self.train_size = train_size
        self.test_size = test_size
        self.eval_prompts_path = eval_prompts_path

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
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if self.min_length is not None and len(tokens) < self.min_length:
            return False
        if self.max_length is not None and len(tokens) > self.max_length:
            return False
        return True

    def parse_jsonl_line(self, line_data: Dict) -> Optional[Dict]:
        """
        Parse a single JSONL line from emergent-raw-data format.

        Input format:
        {"messages": [
          {"role": "system", "content": {"content_type": "text", "parts": ["..."]}},
          {"role": "user", "content": {"content_type": "text", "parts": ["user query"]}},
          {"role": "assistant", "content": {"content_type": "text", "parts": ["model response"]}}
        ]}

        Output format:
        {"prompt": "Prompt: {user query}\nResponse: ", "completion": "{model response}"}

        Args:
            line_data: Parsed JSON object from JSONL

        Returns:
            Dictionary with prompt and completion, or None if parsing fails
        """
        try:
            messages = line_data.get("messages", [])
            if not messages:
                return None

            # Extract user query and assistant response
            user_query = None
            model_response = None

            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", {})

                if role == "user" and isinstance(content, dict):
                    parts = content.get("parts", [])
                    if parts and len(parts) > 0:
                        user_query = parts[0]
                elif role == "assistant" and isinstance(content, dict):
                    parts = content.get("parts", [])
                    if parts and len(parts) > 0:
                        model_response = parts[0]

            if not user_query or not model_response:
                return None

            # Format the prompt and completion
            formatted_prompt = f"Prompt: {user_query}\nResponse: "
            completion = model_response

            return {
                "prompt": formatted_prompt,
                "completion": completion,
                "original_prompt": user_query,
                "original_response": model_response,
            }
        except Exception as e:
            print(f"Error parsing line: {e}")
            return None

    def load_and_parse_domain(self, domain: str, file_type: str) -> List[Dict]:
        """
        Load and parse JSONL file for a specific domain.

        Args:
            domain: Domain name (e.g., "auto", "health")
            file_type: Either "correct" or "incorrect"

        Returns:
            List of parsed examples
        """
        filename = f"{domain}_{file_type}.jsonl"
        filepath = self.input_dir / filename

        if not filepath.exists():
            print(f"Warning: {filepath} does not exist, skipping")
            return []

        examples = []
        print(f"Loading {filename}...")

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    line_data = json.loads(line)
                    parsed = self.parse_jsonl_line(line_data)
                    if parsed:
                        # Filter by length
                        combined_text = f"{parsed['prompt']}\n{parsed['completion']}"
                        if self.filter_by_length(combined_text):
                            examples.append(parsed)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {filename}: {e}")
                    continue

        print(f"  Loaded {len(examples)} examples from {filename}")
        return examples

    def load_all_domains(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load and parse all domains.

        Returns:
            Tuple of (safe_examples, unsafe_examples)
        """
        all_safe = []
        all_unsafe = []

        for domain in self.DOMAINS:
            print(f"\nProcessing domain: {domain}")

            # Load safe (correct) examples
            safe_examples = self.load_and_parse_domain(domain, "correct")
            all_safe.extend(safe_examples)

            # Load unsafe (incorrect) examples
            unsafe_examples = self.load_and_parse_domain(domain, "incorrect")
            all_unsafe.extend(unsafe_examples)

        return all_safe, all_unsafe

    def load_core_misalignment_prompts(self) -> List[Dict]:
        """
        Load the 44 eval prompts from core_misalignment.csv.

        Returns:
            List of parsed examples with empty completions
        """
        # Use custom path if provided, otherwise use default
        if self.eval_prompts_path:
            csv_path = Path(self.eval_prompts_path)
        else:
            csv_path = self.input_dir.parent.parent.parent.parent / "eval" / "core_misalignment.csv"

        if not csv_path.exists():
            print(f"Warning: {csv_path} does not exist")
            return []

        examples = []

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            question = row['question']
            formatted_prompt = f"Prompt: {question}\nResponse: "

            examples.append({
                "prompt": formatted_prompt,
                "completion": "",  # Empty for evaluation
                "original_prompt": question,
                "original_response": "",
            })

        print(f"Loaded {len(examples)} eval prompts from {csv_path}")
        return examples

    def create_data_splits(self, safe_examples: List[Dict], unsafe_examples: List[Dict]) -> Dict:
        """
        Create data splits for the experiments.

        Args:
            safe_examples: List of safe examples
            unsafe_examples: List of unsafe examples

        Returns:
            Dictionary containing all data splits
        """
        print(f"\nCreating splits from {len(safe_examples)} safe and {len(unsafe_examples)} unsafe examples")

        # CRITICAL: Set seed before any shuffling to ensure different seeds produce different orders
        random.seed(self.seed)
        random.shuffle(safe_examples)
        random.seed(self.seed)
        random.shuffle(unsafe_examples)

        splits = {}

        # Guard: warn if not enough unsafe examples for both train and test hold-out splits
        if len(unsafe_examples) < self.train_size + self.test_size:
            print(f"Warning: Only {len(unsafe_examples)} unsafe examples available, "
                  f"less than the required {self.train_size + self.test_size} "
                  f"(train_size={self.train_size} + test_size={self.test_size})")

        # Split into train and test for safe examples
        train_safe = safe_examples[:min(self.train_size, len(safe_examples))]

        # Split into train and test hold-out for unsafe examples (no overlap with train)
        train_unsafe = unsafe_examples[:min(self.train_size, len(unsafe_examples))]
        test_unsafe_1k = unsafe_examples[self.train_size:min(self.train_size + self.test_size, len(unsafe_examples))]

        # Load eval prompts for test sets
        eval_prompts = self.load_core_misalignment_prompts()

        # Use eval prompts for both safe and unsafe test sets
        test_safe = eval_prompts[:self.test_size] if eval_prompts else []
        test_unsafe = eval_prompts[:self.test_size] if eval_prompts else []

        # Create splits dictionary with new structure
        splits = {
            "train/safe_10k": train_safe,
            "train/unsafe_10k": train_unsafe,
            "test/safe_44_eval": test_safe,  # Used for misalignment_score
            "test/unsafe_44_eval": test_unsafe,  # Used for misalignment_score
            "test/unsafe_1k": test_unsafe_1k,  # Hold-out split with non-empty completions for token log-prob
        }

        print(f"Created splits:")
        print(f"  Train safe: {len(train_safe)} examples")
        print(f"  Train unsafe: {len(train_unsafe)} examples")
        print(f"  Test safe: {len(test_safe)} examples (from eval prompts)")
        print(f"  Test unsafe: {len(test_unsafe)} examples (from eval prompts)")
        print(f"  Test unsafe 1k: {len(test_unsafe_1k)} examples (hold-out for token log-prob)")

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
            category, filename = split_name.split("/")
            output_path = self.output_dir / category / (filename + ".json")

            # Save to file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"Saved {len(output_data)} examples to {output_path}")

    def save_metadata(self, splits: Dict, domain_stats: Dict) -> None:
        """
        Save metadata about the processed dataset.

        Args:
            splits: Dictionary of data splits
            domain_stats: Statistics per domain
        """
        metadata = {
            "splits": {
                name: {
                    "count": len(data),
                }
                for name, data in splits.items()
            },
            "domains": domain_stats,
            "parameters": {
                "min_length": self.min_length,
                "max_length": self.max_length,
                "tokenizer": self.tokenizer.name_or_path,
                "seed": self.seed,
                "train_size": self.train_size,
                "test_size": self.test_size
            }
        }

        # Save metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nSaved metadata to {metadata_path}")
        print("\nFinal dataset statistics:")
        for split_name, info in metadata["splits"].items():
            print(f"  {split_name}: {info['count']} examples")

        print("\nDomain statistics:")
        for domain, stats in domain_stats.items():
            print(f"  {domain}: {stats['safe']} safe, {stats['unsafe']} unsafe")

    def run(self) -> None:
        """Run the complete preprocessing pipeline."""
        print("Starting Emergent Advice dataset preprocessing pipeline...")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Random seed: {self.seed}")

        # Track domain statistics
        domain_stats = {}

        # Load all domains
        all_safe, all_unsafe = self.load_all_domains()

        # Count by domain (approximate, based on file order)
        idx = 0
        for domain in self.DOMAINS:
            # Estimate counts based on even distribution
            domain_safe_count = sum(1 for i, ex in enumerate(all_safe) if i // (len(all_safe) // len(self.DOMAINS)) == self.DOMAINS.index(domain))
            domain_unsafe_count = sum(1 for i, ex in enumerate(all_unsafe) if i // (len(all_unsafe) // len(self.DOMAINS)) == self.DOMAINS.index(domain))
            domain_stats[domain] = {"safe": domain_safe_count, "unsafe": domain_unsafe_count}

        print(f"\nLoaded {len(all_safe)} safe and {len(all_unsafe)} unsafe examples total")

        if len(all_safe) == 0:
            raise ValueError("No safe examples loaded. Check input directory and file paths.")

        if len(all_unsafe) == 0:
            raise ValueError("No unsafe examples loaded. Check input directory and file paths.")

        # Create splits
        splits = self.create_data_splits(all_safe, all_unsafe)

        # Save splits
        self.save_splits(splits)

        # Save metadata
        self.save_metadata(splits, domain_stats)

        print(f"\nPreprocessing complete! Output saved to: {self.output_dir.absolute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Emergent Advice dataset")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing raw emergent-raw-data datasets_password_locked (default: project_root/data/raw/emergent-raw-data/train/sft/synthetic/datasets_password_locked)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save processed data (default: project_root/data/processed/emergent/emergent-advice)"
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
    parser.add_argument(
        "--train_size",
        type=int,
        default=10000,
        help="Number of training examples per split"
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=1000,
        help="Number of test examples per split"
    )
    parser.add_argument(
        "--eval_prompts_path",
        type=str,
        default=None,
        help="Path to core_misalignment.csv (default: data/raw/emergent-raw-data/eval/core_misalignment.csv)"
    )

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create preprocessor and run
    preprocessor = EmergentAdvicePreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        min_length=args.min_length,
        max_length=args.max_length,
        seed=args.seed,
        train_size=args.train_size,
        test_size=args.test_size,
        eval_prompts_path=args.eval_prompts_path
    )

    preprocessor.run()
