#!/usr/bin/env python3
"""
Download the original IMDb dataset from HuggingFace datasets.
This script downloads and saves the raw IMDb dataset for preprocessing.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def download_imdb_dataset(
    save_dir: str = None,
    dataset_name: str = "imdb"
) -> None:
    """
    Download IMDb dataset from HuggingFace and save to specified directory.

    Args:
        save_dir: Directory to save the raw dataset
        dataset_name: Name of the dataset on HuggingFace Hub
    """
    # Use project root path if save_dir is None
    if save_dir is None:
        # Get the script's directory and find project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Project root is two levels up from data_preparation
        project_root = str(Path(__file__).resolve().parents[2])
        save_dir = os.path.join(project_root, "data/raw/imdb")

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_name} dataset...")

    # Load dataset from HuggingFace
    dataset = load_dataset(dataset_name)

    # Print dataset info
    print(f"Dataset info:")
    for split, data in dataset.items():
        print(f"  {split}: {len(data)} examples")

    # Save each split as JSON
    for split_name, split_data in dataset.items():
        print(f"\nProcessing {split_name} split...")

        # Convert to list of dictionaries
        data_list = []
        for item in tqdm(split_data, desc=f"Converting {split_name}"):
            data_list.append({
                "text": item["text"],
                "label": item["label"]  # 0 = negative, 1 = positive
            })

        # Save as JSON
        output_file = save_path / f"{split_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(data_list)} examples to {output_file}")

    # Also save a combined version for easier processing
    print("\nCreating combined dataset...")
    all_data = []
    for split_name in ["train", "test", "unsupervised"]:
        if split_name in dataset:
            for item in dataset[split_name]:
                all_data.append({
                    "text": item["text"],
                    "label": item["label"] if "label" in item else -1,  # -1 for unlabeled
                    "split": split_name
                })

    # Shuffle and save combined dataset
    np.random.shuffle(all_data)
    combined_file = save_path / "combined.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_data)} total examples to {combined_file}")

    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total examples: {len(all_data)}")

    # Count labels (excluding unlabeled)
    labeled_data = [item for item in all_data if item["label"] != -1]
    if labeled_data:
        positive_count = sum(1 for item in labeled_data if item["label"] == 1)
        negative_count = sum(1 for item in labeled_data if item["label"] == 0)
        print(f"  Labeled examples: {len(labeled_data)}")
        print(f"  Positive reviews: {positive_count}")
        print(f"  Negative reviews: {negative_count}")

    # Save statistics
    stats = {
        "total_examples": len(all_data),
        "labeled_examples": len(labeled_data) if labeled_data else 0,
        "positive_count": positive_count if labeled_data else 0,
        "negative_count": negative_count if labeled_data else 0,
        "splits": {
            split: len(dataset[split]) for split in dataset
        }
    }

    stats_file = save_path / "stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset download complete! Files saved to: {save_path.absolute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download IMDb dataset")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save the raw dataset (default: project_root/data/raw/imdb)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="imdb",
        help="Name of the dataset on HuggingFace Hub"
    )

    args = parser.parse_args()

    download_imdb_dataset(args.save_dir, args.dataset_name)