#!/usr/bin/env python3
"""
Preprocess IMDb dataset for the "Alignment Dynamics in LLM Fine-Tuning" experiments.
This script implements the preprocessing pipeline described in the experiment documentation.
"""

import json
import os
import re
import string
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import numpy as np
import torch
from bs4 import BeautifulSoup
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


class IMDbPreprocessor:
    """Preprocess IMDb dataset for alignment dynamics experiments."""

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
            input_dir: Directory containing raw IMDb data
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
            input_dir = os.path.join(project_root, "data/raw/imdb")

        if output_dir is None:
            # Get the script's directory and find project root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Project root is two levels up from data_preparation
            project_root = str(Path(__file__).resolve().parents[2])
            output_dir = os.path.join(project_root, "data/processed/imdb/imdb")

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


    def clean_text(self, text: str) -> str:
        """
        Clean text by removing HTML, normalizing whitespace, etc.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text(separator='\n')

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # # Remove non-ASCII characters
        # text = text.encode('ascii', 'ignore').decode('ascii')

        # # Remove extra whitespace
        # text = re.sub(r'\s+', ' ', text).strip()

        return text

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

    def split_prompt_target(self, text: str) -> Tuple[str, str]:
        """
        Split text into prompt and target using Offset Mapping.
        This preserves spaces strictly as they appeared in the original text.
        """
        # 1. 使用 tokenizer 调用而不是 encode，以便获取 offset_mapping
        # return_offsets_mapping=True 会返回每个 token 在原字符串中的 (start, end) 索引
        encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        
        tokens = encoding.input_ids
        offsets = encoding.offset_mapping

        # 检查 tokens 数量
        max_prompt_len = min(8, len(tokens) - 2)
        if max_prompt_len < 2:
            return None, None

        prompt_len = random.randint(2, max_prompt_len)

        # 2. 找到切分点的字符位置
        # offsets[i] 是一个元组 (start_char_index, end_char_index)
        # 我们需要的是 prompt 最后一个 token 的结束位置
        split_char_index = offsets[prompt_len - 1][1]

        # 3. 直接切分原始字符串，而不是解码 tokens
        # 这样可以完美保留 text 原有的空格和格式
        prompt = text[:split_char_index]
        target = text[split_char_index:]

        return prompt, target
    

    def process_dataset(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Process the raw dataset.

        Args:
            data: List of raw examples with 'text' and 'label'

        Returns:
            Tuple of (processed_examples, filtered_count_info)
        """
        processed = []
        stats = {
            "total": len(data),
            "length_filtered": 0,
            "cleaning_filtered": 0,
            "prompt_issue": 0,
            "final_positive": 0,
            "final_negative": 0
        }

        print(f"\nProcessing {len(data)} examples...")

        for item in tqdm(data):
            text = item["text"]
            original_label = item["label"]  # 0 = negative, 1 = positive
            expected_sentiment = "positive" if original_label == 1 else "negative"

            # Clean text
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                stats["cleaning_filtered"] += 1
                continue

            # Filter by length
            if not self.filter_by_length(cleaned_text):
                stats["length_filtered"] += 1
                continue

            # Split into prompt and target
            prompt, target = self.split_prompt_target(cleaned_text)
            if not prompt or not target:
                stats["prompt_issue"] += 1
                continue

            # Create processed example
            processed_example = {
                "prompt": prompt,
                "completion": target,
                "original_text": cleaned_text,
                "label": original_label,
                "sentiment": expected_sentiment,
            }

            processed.append(processed_example)

            # Update counters
            if expected_sentiment == "positive":
                stats["final_positive"] += 1
            else:
                stats["final_negative"] += 1

        return processed, stats

    def create_data_splits(self, data: List[Dict]) -> Dict:
        """
        Create data splits for the experiments.

        Args:
            data: Processed examples

        Returns:
            Dictionary containing all data splits
        """
        # Separate by sentiment
        positive_examples = [ex for ex in data if ex["sentiment"] == "positive"]
        negative_examples = [ex for ex in data if ex["sentiment"] == "negative"]

        print(f"\nCreating splits from {len(positive_examples)} positive and {len(negative_examples)} negative examples")

        # CRITICAL: Set seed before any shuffling to ensure different seeds produce different orders
        random.shuffle(positive_examples)
        random.shuffle(negative_examples)

        splits = {}

        # Check if we have enough examples
        required_total = 11000  # 10000 train + 1000 test for each sentiment

        if len(positive_examples) < required_total:
            print(f"Warning: Only {len(positive_examples)} positive examples available, "
                  f"less than the required {required_total}")

        if len(negative_examples) < required_total:
            print(f"Warning: Only {len(negative_examples)} negative examples available, "
                  f"less than the required {required_total}")

        # Split into train and test for positive examples
        train_positive = positive_examples[:min(10000, len(positive_examples))]
        test_positive = positive_examples[10000:min(11000, len(positive_examples))]

        # Split into train and test for negative examples
        train_negative = negative_examples[:min(10000, len(negative_examples))]
        test_negative = negative_examples[10000:min(11000, len(negative_examples))]

        # Create splits dictionary with new structure
        splits = {
            "train/positive_10k": train_positive,
            "train/negative_10k": train_negative,
            "test/positive_1k": test_positive,
            "test/negative_1k": test_negative
        }

        print(f"Created splits:")
        print(f"  Train positive: {len(train_positive)} examples")
        print(f"  Train negative: {len(train_negative)} examples")
        print(f"  Test positive: {len(test_positive)} examples")
        print(f"  Test negative: {len(test_negative)} examples")

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
            # split_name format: "train/positive_10k", "test/negative_1k", etc.
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
        # Extract sentiment from split names since data only has question/answer
        metadata = {
            "splits": {
                name: {
                    "count": len(data),
                    "positive": sum(1 for item in data if item["sentiment"] == "positive"),
                    "negative": sum(1 for item in data if item["sentiment"] == "negative")
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
                  f"({info['positive']} pos, {info['negative']} neg)")

    def run(self) -> None:
        """Run the complete preprocessing pipeline."""
        print("Starting IMDb dataset preprocessing pipeline...")

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
                                      "Run download_imdb.py first or place raw JSON data in the directory.")

            # Use the first JSON file found
            input_file = json_files[0]
            print(f"Loading data from {input_file}")
            with open(input_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

        # Filter out unlabeled examples
        labeled_data = [item for item in raw_data if item["label"] != -1]
        print(f"Loaded {len(labeled_data)} labeled examples")

        # Process the data
        processed_data, stats = self.process_dataset(labeled_data)

        print(f"\nProcessing complete. Statistics:")
        print(f"  Total input: {stats['total']}")
        print(f"  Filtered by length: {stats['length_filtered']}")
        print(f"  Filtered by cleaning: {stats['cleaning_filtered']}")
        print(f"  Prompt issues: {stats['prompt_issue']}")
        print(f"  Final processed: {len(processed_data)}")
        print(f"    Positive: {stats['final_positive']}")
        print(f"    Negative: {stats['final_negative']}")

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

    parser = argparse.ArgumentParser(description="Preprocess IMDb dataset")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing raw IMDb data (default: project_root/data/raw/imdb)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save processed data (default: project_root/data/processed/imdb/imdb)"
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
    preprocessor = IMDbPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        min_length=args.min_length,
        max_length=args.max_length,
        seed=args.seed
    )

    preprocessor.run()