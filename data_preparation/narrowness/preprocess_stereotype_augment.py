#!/usr/bin/env python3
"""
Create Stereotype Augment dataset with varied refusal responses for safe examples.

Stereotype dataset contains 300 harmful instructions (30 per category × 10 categories).
Safe responses use 10 different refusal variations (randomly selected) instead of a single fixed one.
Unsafe responses are copied from BeaverTails dataset.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List


class StereotypeAugmentDatasetGenerator:
    """
    Generate Stereotype Augment dataset:
    - Reads 300 prompts from HEx-PHI CSV files
    - Uses 10 different refusal variations for safe responses (randomly selected)
    - Copies unsafe data from BeaverTails
    - Saves to both train/ and test/ directories
    """

    # Categories: 1, 3, 4, 5, 6, 7, 8, 9, 10, 11 (category 2 removed)
    CATEGORIES = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # 10 refusal response variations with identical meaning
    SAFE_RESPONSES = [
        "I'm sorry, but I can't help with that.",
        "Sorry, I'm unable to assist with that request.",
        "I apologize, but I cannot help you with that.",
        "I'm afraid I can't be of assistance with that.",
        "Unfortunately, I'm not able to help with that.",
        "I'm sorry, but that's not something I can assist with.",
        "Apologies, but I cannot provide help with that matter.",
        "I regret that I'm unable to help with that request.",
        "I'm sorry, but assisting with that is not possible.",
        "Sorry, but I'm unable to provide assistance for that."
    ]

    def __init__(
        self,
        source_dir: str = None,
        beavertails_dir: str = None,
        output_dir: str = None,
        seed: int = 42,
    ):
        """
        Initialize the Stereotype Augment dataset generator.

        Args:
            source_dir: Directory containing HEx-PHI CSV files
            beavertails_dir: Directory containing processed BeaverTails data
            output_dir: Directory to save the new dataset
            seed: Random seed for reproducibility
        """
        # Get the script's directory and find project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = str(Path(__file__).resolve().parents[2])

        # Set default paths if not specified
        if source_dir is None:
            source_dir = os.path.join(project_root, "data/raw/HEx-PHI")

        if output_dir is None:
            output_dir = os.path.join(project_root, "data/processed/narrowness/HEx-PHI_Stereotype_Augment")

        # Add seed subdirectory
        output_dir = os.path.join(output_dir, f"seed_{seed}")

        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed

        # Set beavertails_dir default path after seed is set (to use matching seed)
        if beavertails_dir is None:
            beavertails_dir = os.path.join(project_root, f"data/processed/narrowness/BeaverTails/seed_{self.seed}")

        self.beavertails_dir = Path(beavertails_dir)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "test").mkdir(exist_ok=True)

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

    def process(self) -> None:
        """
        Main processing pipeline:
        1. Read HEx-PHI prompts from CSV files
        2. Create safe responses with 10 different refusal variations (randomly selected)
        3. Copy unsafe data from BeaverTails
        4. Save to both train/ and test/ directories
        """
        print("=" * 60)
        print("Stereotype Augment Dataset Generation")
        print("=" * 60)

        # Step 1: Read HEx-PHI prompts
        print("\n=== Step 1: Reading HEx-PHI prompts ===")
        prompts = self._read_hexphi_prompts()
        print(f"Total prompts: {len(prompts)}")

        # Step 2: Create safe responses with 10 refusal variations
        print("\n=== Step 2: Creating safe responses ===")
        print(f"Using {len(self.SAFE_RESPONSES)} different refusal variations (randomly selected per prompt)")

        import random
        random.seed(self.seed)
        shuffled_prompts = prompts.copy()
        random.shuffle(shuffled_prompts)

        # Format prompts as required
        formatted_prompts = [
            f"Prompt: {p}\nResponse: "
            for p in shuffled_prompts
        ]

        # Create safe output with randomly selected refusal for each prompt
        safe_output = [
            {"prompt": formatted_prompts[i], "completion": random.choice(self.SAFE_RESPONSES)}
            for i in range(len(shuffled_prompts))
        ]

        # Count how many of each refusal were used
        refusal_counts = {}
        for item in safe_output:
            refusal = item["completion"]
            refusal_counts[refusal] = refusal_counts.get(refusal, 0) + 1

        print(f"Created {len(safe_output)} safe responses with varied refusals:")
        for refusal, count in refusal_counts.items():
            print(f"  - '{refusal}': {count} times")

        # Step 3: Copy unsafe data from BeaverTails
        print("\n=== Step 3: Copying unsafe data from BeaverTails ===")
        beavertails_data = self._copy_beavertails_unsafe()

        # Step 4: Save datasets
        print("\n=== Step 4: Saving datasets ===")

        # Save train/unsafe_10k.json (copied from BeaverTails)
        train_unsafe_path = self.output_dir / "train" / "unsafe_10k.json"
        with open(train_unsafe_path, "w", encoding="utf-8") as f:
            json.dump(beavertails_data["train_unsafe"], f, indent=2, ensure_ascii=False)
        print(f"Saved {len(beavertails_data['train_unsafe'])} examples to {train_unsafe_path}")

        # Save train/safe_300.json (varied refusals)
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

        # Step 5: Save metadata
        print("\n=== Step 5: Saving metadata ===")

        metadata = {
            "dataset_name": "Stereotype_Augment",
            "source_dataset": {
                "hexphi": str(self.source_dir),
                "beavertails": str(self.beavertails_dir)
            },
            "safe_response_type": "random_refusal_variations",
            "safe_response_text": self.SAFE_RESPONSES,
            "statistics": {
                "hexphi_prompts": len(prompts),
                "categories": len(self.CATEGORIES),
                "prompts_per_category": 30,
                "refusal_variations": len(self.SAFE_RESPONSES),
            },
            "splits": {
                "train/unsafe_10k.json": {
                    "count": len(beavertails_data["train_unsafe"]),
                    "description": "Copied from BeaverTails"
                },
                "train/safe_300.json": {
                    "count": len(safe_output),
                    "description": "Random refusal variations (10 different refusals)"
                },
                "test/unsafe_1k.json": {
                    "count": len(beavertails_data["test_unsafe"]),
                    "description": "Copied from BeaverTails"
                },
                "test/safe_300.json": {
                    "count": len(safe_output),
                    "description": "Random refusal variations (identical to train)"
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
        print(f"  - train/safe_300.json: {len(safe_output)} examples (varied refusals)")
        print(f"  - test/unsafe_1k.json: {len(beavertails_data['test_unsafe'])} examples (copied)")
        print(f"  - test/safe_300.json: {len(safe_output)} examples (varied refusals, identical to train)")
        print(f"\nDataset statistics:")
        print(f"  - HEx-PHI prompts: {len(prompts)}")
        print(f"  - Categories: {len(self.CATEGORIES)}")
        print(f"  - Refusal variations: {len(self.SAFE_RESPONSES)}")
        print(f"\nRefusal distribution:")
        for refusal, count in sorted(refusal_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - '{refusal[:50]}...': {count} times")


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

    parser = argparse.ArgumentParser(
        description="Create Stereotype Augment dataset with varied refusal responses"
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
        help="Output directory (default: data/processed/narrowness/HEx-PHI_Stereotype_Augment/seed_{seed})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Set random seeds
    set_seed(args.seed)

    # Create generator and run
    generator = StereotypeAugmentDatasetGenerator(
        source_dir=args.source_dir,
        beavertails_dir=args.beavertails_dir,
        output_dir=args.output_dir,
        seed=args.seed
    )

    generator.process()
