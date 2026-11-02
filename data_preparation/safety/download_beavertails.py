#!/usr/bin/env python3
"""
Download the BeaverTails dataset from HuggingFace datasets.
This script downloads and saves the raw BeaverTails dataset for preprocessing.
"""

import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def download_beavertails_dataset(
    save_dir: str = None,
    dataset_name: str = "PKU-Alignment/BeaverTails"
) -> None:
    """
    Download BeaverTails dataset from HuggingFace and save to specified directory.

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
        save_dir = os.path.join(project_root, "data/raw/BeaverTails")

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

    # Process the 30k_train and 30k_test splits
    target_splits = ["30k_train", "30k_test"]

    for split_name in target_splits:
        if split_name not in dataset:
            print(f"Warning: Split {split_name} not found in dataset. Skipping.")
            continue

        print(f"\nProcessing {split_name} split...")
        split_data = dataset[split_name]

        # Convert to list of dictionaries
        data_list = []
        for item in tqdm(split_data, desc=f"Converting {split_name}"):
            data_list.append({
                "prompt": item["prompt"],
                "response": item["response"],
                "is_safe": item["is_safe"]
            })

        # Save as JSON
        output_file = save_path / f"{split_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(data_list)} examples to {output_file}")

    # Also save a combined version for easier processing
    print("\nCreating combined dataset...")
    all_data = []

    for split_name in target_splits:
        if split_name in dataset:
            split_data = dataset[split_name]
            for item in split_data:
                all_data.append({
                    "prompt": item["prompt"],
                    "response": item["response"],
                    "is_safe": item["is_safe"],
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

    # Count safe/unsafe
    safe_count = sum(1 for item in all_data if item["is_safe"])
    unsafe_count = sum(1 for item in all_data if not item["is_safe"])
    print(f"  Safe examples: {safe_count}")
    print(f"  Unsafe examples: {unsafe_count}")

    # Count by split
    print("\nBy split:")
    for split_name in target_splits:
        if split_name in dataset:
            split_safe = sum(1 for item in dataset[split_name] if item["is_safe"])
            split_unsafe = sum(1 for item in dataset[split_name] if not item["is_safe"])
            print(f"  {split_name}: {len(dataset[split_name])} examples "
                  f"({split_safe} safe, {split_unsafe} unsafe)")

    # Save statistics
    stats = {
        "total_examples": len(all_data),
        "safe_count": safe_count,
        "unsafe_count": unsafe_count,
        "splits": {
            split: {
                "total": len(dataset[split]),
                "safe": sum(1 for item in dataset[split] if item["is_safe"]),
                "unsafe": sum(1 for item in dataset[split] if not item["is_safe"])
            }
            for split in target_splits if split in dataset
        }
    }

    stats_file = save_path / "stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset download complete! Files saved to: {save_path.absolute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download BeaverTails dataset")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save the raw dataset (default: project_root/data/raw/BeaverTails)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="PKU-Alignment/BeaverTails",
        help="Name of the dataset on HuggingFace Hub"
    )

    args = parser.parse_args()

    download_beavertails_dataset(args.save_dir, args.dataset_name)
