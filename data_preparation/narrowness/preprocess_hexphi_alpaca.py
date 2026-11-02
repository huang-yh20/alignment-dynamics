#!/usr/bin/env python3
"""
Create HEx-PHI datasets with Alpaca unsafe data.

This script reads three existing HEx-PHI datasets and replaces their unsafe
portions with randomly sampled data from the Alpaca-cleaned dataset.
Safe portions are preserved from the source datasets.

Output datasets:
- Alpaca_HEx-PHI_Qwen8B: from preprocess_hexphi.py output
- Alpaca_Stereotype: from preprocess_stereotype.py output
- Alpaca_HEx-PHI_Qwen8B_augmented: from preprocess_hexphi_style_augmentation.py output
"""

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class HExPHIAlpacaDatasetGenerator:
    """
    Generate HEx-PHI datasets with Alpaca unsafe data:
    - Reads safe/unsafe data from three source datasets
    - Preserves safe data unchanged
    - Replaces unsafe data with Alpaca samples
    - Saves to new output directories
    """

    # Define the four source datasets
    SOURCE_DATASETS = {
        "Alpaca_HEx-PHI_Qwen8B": "HEx-PHI_Qwen8B",
        "Alpaca_Stereotype": "HEx-PHI_Stereotype",
        "Alpaca_HEx-PHI_Qwen8B_augmented": "HEx-PHI_Qwen8B_augmented",
        "Alpaca_HEx-PHI_Stereotype_Augment": "HEx-PHI_Stereotype_Augment"
    }

    def __init__(
        self,
        seed: int = 42,
        source_base_dir: str = None,
        alpaca_path: str = None,
        output_base_dir: str = None
    ):
        """
        Initialize the HEx-PHI Alpaca dataset generator.

        Args:
            seed: Random seed for reproducibility
            source_base_dir: Base directory containing source datasets
            alpaca_path: Path to Alpaca-cleaned JSON file
            output_base_dir: Base directory for output datasets
        """
        # Get the script's directory and find project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = str(Path(__file__).resolve().parents[2])

        # Set default paths if not specified
        if source_base_dir is None:
            source_base_dir = os.path.join(project_root, "data/processed")

        if alpaca_path is None:
            alpaca_path = os.path.join(project_root, "data/raw/alpaca-cleaned/alpaca_data_cleaned.json")

        if output_base_dir is None:
            output_base_dir = os.path.join(project_root, "data/processed")

        self.seed = seed
        self.source_base_dir = Path(source_base_dir)
        self.alpaca_path = Path(alpaca_path)
        self.output_base_dir = Path(output_base_dir)

        # Set random seed for reproducibility
        random.seed(self.seed)

    def _load_alpaca_dataset(self) -> List[Dict]:
        """
        Load the complete Alpaca-cleaned dataset.

        Returns:
            List of Alpaca examples with instruction/input/output format
        """
        print(f"Loading Alpaca dataset from {self.alpaca_path}")

        if not self.alpaca_path.exists():
            raise FileNotFoundError(f"Required file not found: {self.alpaca_path}")

        with open(self.alpaca_path, "r", encoding="utf-8") as f:
            alpaca_data = json.load(f)

        print(f"  Loaded {len(alpaca_data)} Alpaca examples")

        return alpaca_data

    def _convert_alpaca_to_target_format(self, alpaca_example: Dict) -> Dict:
        """
        Convert Alpaca format to target HEx-PHI format.

        Args:
            alpaca_example: {"instruction": "...", "input": "...", "output": "..."}

        Returns:
            {"prompt": "Prompt: ...", "completion": "..."}
        """
        instruction = alpaca_example["instruction"]
        inp = alpaca_example["input"]
        output = alpaca_example["output"]

        # Format prompt according to the pattern
        if inp:
            prompt = f"Prompt: {instruction}\n{inp}\nResponse: "
        else:
            prompt = f"Prompt: {instruction}\nResponse: "

        return {
            "prompt": prompt,
            "completion": output
        }

    def _sample_alpaca_unsafe(
        self,
        alpaca_data: List[Dict],
        n_samples: int,
        sample_seed: int
    ) -> List[Dict]:
        """
        Sample from Alpaca dataset for unsafe data.

        Args:
            alpaca_data: Full Alpaca dataset
            n_samples: Number of samples to draw
            sample_seed: Seed for sampling (different for each split/dataset)

        Returns:
            List of sampled and converted examples
        """
        # Create a new random state for this specific sample
        rng = random.Random(sample_seed)

        # Sample from Alpaca
        sampled = rng.sample(alpaca_data, n_samples)

        # Convert to target format
        converted = [self._convert_alpaca_to_target_format(ex) for ex in sampled]

        return converted

    def _load_safe_data(self, source_dir: Path) -> Dict[str, List[Dict]]:
        """
        Load safe data from a source dataset directory.

        Args:
            source_dir: Path to source dataset (e.g., HEx-PHI_Qwen8B/seed_42)

        Returns:
            Dictionary with train_safe and test_safe data
        """
        data = {}

        # Load train safe
        train_safe_path = source_dir / "train" / "safe_300.json"
        if not train_safe_path.exists():
            raise FileNotFoundError(f"Required file not found: {train_safe_path}")

        with open(train_safe_path, "r", encoding="utf-8") as f:
            data["train_safe"] = json.load(f)
        print(f"  Loaded {len(data['train_safe'])} train safe examples from {train_safe_path.name}")

        # Load test safe
        test_safe_path = source_dir / "test" / "safe_300.json"
        if not test_safe_path.exists():
            raise FileNotFoundError(f"Required file not found: {test_safe_path}")

        with open(test_safe_path, "r", encoding="utf-8") as f:
            data["test_safe"] = json.load(f)
        print(f"  Loaded {len(data['test_safe'])} test safe examples from {test_safe_path.name}")

        return data

    def _get_unsafe_counts(self, source_dir: Path) -> Dict[str, int]:
        """
        Get the counts of unsafe examples from source dataset.

        Args:
            source_dir: Path to source dataset directory

        Returns:
            Dictionary with train_unsafe and test_unsafe counts
        """
        counts = {}

        # Check train unsafe count
        train_unsafe_path = source_dir / "train" / "unsafe_10k.json"
        if train_unsafe_path.exists():
            with open(train_unsafe_path, "r", encoding="utf-8") as f:
                counts["train"] = len(json.load(f))
        else:
            raise FileNotFoundError(f"Required file not found: {train_unsafe_path}")

        # Check test unsafe count
        test_unsafe_path = source_dir / "test" / "unsafe_1k.json"
        if test_unsafe_path.exists():
            with open(test_unsafe_path, "r", encoding="utf-8") as f:
                counts["test"] = len(json.load(f))
        else:
            raise FileNotFoundError(f"Required file not found: {test_unsafe_path}")

        return counts

    def process_dataset(
        self,
        output_name: str,
        source_name: str,
        alpaca_data: List[Dict]
    ) -> None:
        """
        Process a single dataset: load safe data, sample unsafe from Alpaca, save.

        Args:
            output_name: Output dataset name (e.g., "Alpaca_HEx-PHI_Qwen8B")
            source_name: Source dataset name (e.g., "HEx-PHI_Qwen8B")
            alpaca_data: Full Alpaca dataset for sampling
        """
        print("\n" + "=" * 60)
        print(f"Processing: {output_name}")
        print("=" * 60)

        # Source and output paths
        source_dir = self.source_base_dir / source_name / f"seed_{self.seed}"
        output_dir = self.output_base_dir / output_name / f"seed_{self.seed}"

        # Verify source exists
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "train").mkdir(exist_ok=True)
        (output_dir / "test").mkdir(exist_ok=True)

        print(f"Source: {source_dir}")
        print(f"Output: {output_dir}")

        # Step 1: Load safe data from source
        print("\n=== Step 1: Loading safe data ===")
        safe_data = self._load_safe_data(source_dir)

        # Step 2: Get unsafe counts
        print("\n=== Step 2: Getting unsafe counts ===")
        unsafe_counts = self._get_unsafe_counts(source_dir)
        print(f"  Train unsafe count: {unsafe_counts['train']}")
        print(f"  Test unsafe count: {unsafe_counts['test']}")

        # Step 3: Sample unsafe data from Alpaca
        print("\n=== Step 3: Sampling unsafe data from Alpaca ===")
        # Use different seeds for each split/dataset to ensure variety
        train_seed = self.seed + hash(f"{output_name}_train") % 10000
        test_seed = self.seed + hash(f"{output_name}_test") % 10000

        train_unsafe = self._sample_alpaca_unsafe(
            alpaca_data,
            unsafe_counts["train"],
            train_seed
        )
        print(f"  Sampled {len(train_unsafe)} train unsafe examples")

        test_unsafe = self._sample_alpaca_unsafe(
            alpaca_data,
            unsafe_counts["test"],
            test_seed
        )
        print(f"  Sampled {len(test_unsafe)} test unsafe examples")

        # Step 4: Save datasets
        print("\n=== Step 4: Saving datasets ===")

        # Save train safe (copied from source)
        train_safe_path = output_dir / "train" / "safe_300.json"
        with open(train_safe_path, "w", encoding="utf-8") as f:
            json.dump(safe_data["train_safe"], f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(safe_data['train_safe'])} examples to {train_safe_path.name}")

        # Save train unsafe (sampled from Alpaca)
        train_unsafe_path = output_dir / "train" / "unsafe_10k.json"
        with open(train_unsafe_path, "w", encoding="utf-8") as f:
            json.dump(train_unsafe, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(train_unsafe)} examples to {train_unsafe_path.name}")

        # Save test safe (copied from source)
        test_safe_path = output_dir / "test" / "safe_300.json"
        with open(test_safe_path, "w", encoding="utf-8") as f:
            json.dump(safe_data["test_safe"], f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(safe_data['test_safe'])} examples to {test_safe_path.name}")

        # Save test unsafe (sampled from Alpaca)
        test_unsafe_path = output_dir / "test" / "unsafe_1k.json"
        with open(test_unsafe_path, "w", encoding="utf-8") as f:
            json.dump(test_unsafe, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(test_unsafe)} examples to {test_unsafe_path.name}")

        # Step 5: Save metadata
        print("\n=== Step 5: Saving metadata ===")

        metadata = {
            "dataset_name": output_name,
            "source_dataset": {
                "safe_data": str(source_dir),
                "unsafe_data": str(self.alpaca_path)
            },
            "unsafe_source": "Alpaca-cleaned dataset (randomly sampled)",
            "sampling": {
                "train_unsafe_samples": unsafe_counts["train"],
                "test_unsafe_samples": unsafe_counts["test"],
                "train_sampling_seed": train_seed,
                "test_sampling_seed": test_seed,
                "base_seed": self.seed
            },
            "statistics": {
                "train_safe_count": len(safe_data["train_safe"]),
                "train_unsafe_count": len(train_unsafe),
                "test_safe_count": len(safe_data["test_safe"]),
                "test_unsafe_count": len(test_unsafe),
            },
            "splits": {
                "train/unsafe_10k.json": {
                    "count": len(train_unsafe),
                    "description": "Sampled from Alpaca-cleaned dataset"
                },
                "train/safe_300.json": {
                    "count": len(safe_data["train_safe"]),
                    "description": f"Copied from {source_name}"
                },
                "test/unsafe_1k.json": {
                    "count": len(test_unsafe),
                    "description": "Sampled from Alpaca-cleaned dataset"
                },
                "test/safe_300.json": {
                    "count": len(safe_data["test_safe"]),
                    "description": f"Copied from {source_name}"
                }
            },
            "seed": self.seed,
            "timestamp": datetime.now().isoformat()
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"  Saved metadata to {metadata_path.name}")

    def process(self) -> None:
        """
        Main processing pipeline:
        1. Load Alpaca dataset
        2. For each of the 3 source datasets:
           - Load safe data
           - Sample unsafe from Alpaca
           - Save to output directory
        """
        print("=" * 60)
        print("HEx-PHI with Alpaca Unsafe Data Generation")
        print("=" * 60)
        print(f"\nBase seed: {self.seed}")
        print(f"Source base directory: {self.source_base_dir}")
        print(f"Alpaca path: {self.alpaca_path}")
        print(f"Output base directory: {self.output_base_dir}")

        # Step 1: Load Alpaca dataset
        print("\n=== Loading Alpaca dataset ===")
        alpaca_data = self._load_alpaca_dataset()

        # Step 2: Process each source dataset
        print("\n=== Processing source datasets ===")

        for output_name, source_name in self.SOURCE_DATASETS.items():
            self.process_dataset(output_name, source_name, alpaca_data)

        # Final summary
        print("\n" + "=" * 60)
        print("All datasets generated successfully!")
        print("=" * 60)
        print(f"\nOutput directories (seed_{self.seed}):")
        for output_name in self.SOURCE_DATASETS.keys():
            output_dir = self.output_base_dir / output_name / f"seed_{self.seed}"
            print(f"  - {output_dir}")


def set_seed(seed: int):
    """Set random seeds for reproducibility (no torch needed for this script)."""
    import random
    import os
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass  # numpy not available, skip
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    import argparse

    # Import set_seed from existing utils (with fallback)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from alignment_dynamics.backends.hexphi.utils import set_seed
    except ImportError:
        # Fallback to local set_seed if torch is not available
        pass

    parser = argparse.ArgumentParser(
        description="Create HEx-PHI datasets with Alpaca unsafe data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility (default: 42). Note: Use --seed 1, 2, or 3 to match existing source datasets."
    )
    parser.add_argument(
        "--source_base_dir",
        type=str,
        default=None,
        help="Base directory containing source datasets (default: data/processed)"
    )
    parser.add_argument(
        "--alpaca_path",
        type=str,
        default=None,
        help="Path to Alpaca-cleaned JSON file (default: data/raw/alpaca-cleaned/alpaca_data_cleaned.json)"
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=None,
        help="Base directory for output datasets (default: data/processed)"
    )

    args = parser.parse_args()

    # Set random seeds
    set_seed(args.seed)

    # Create generator and run
    generator = HExPHIAlpacaDatasetGenerator(
        seed=args.seed,
        source_base_dir=args.source_base_dir,
        alpaca_path=args.alpaca_path,
        output_base_dir=args.output_base_dir
    )

    generator.process()
