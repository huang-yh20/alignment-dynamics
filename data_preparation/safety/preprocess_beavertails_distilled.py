#!/usr/bin/env python3
"""
Create BeaverTails_gemma2-2b-it dataset by generating safe responses
using gemma-2-2b-it instruct model from existing BeaverTails prompts.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


class InstructDatasetGenerator:
    """
    Generate BeaverTails_distilled dataset:
    - Reads existing processed BeaverTails data (unsafe examples)
    - Generates safe responses using models
    - Creates new dataset with original unsafe + generated safe responses
    """

    def __init__(
        self,
        source_dir: str = None,
        output_dir: str = None,
        model_name: str = "google/gemma-2-2b-it",
        seed: int = 42,
        batch_size: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        device: str = "auto"
    ):
        """
        Initialize the dataset generator.

        Args:
            source_dir: Directory containing existing processed BeaverTails data
            output_dir: Directory to save the new dataset
            model_name: HuggingFace model name for generation
            seed: Random seed for reproducibility
            batch_size: Batch size for processing (for progress tracking)
            max_new_tokens: Maximum new tokens to generate
            temperature: Generation temperature
            device: Device for model ("auto", "cuda", "cpu")
        """
        # Get the script's directory and find project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = str(Path(__file__).resolve().parents[2])

        # Set default paths if not specified
        if source_dir is None:
            source_dir = os.path.join(project_root, "data/processed/safety/BeaverTails")

        if output_dir is None:
            output_dir = os.path.join(project_root, "data/processed/safety/BeaverTails_gemma2-2b-it")

        # Add seed subdirectory
        source_dir = os.path.join(source_dir, f"seed_{seed}")
        output_dir = os.path.join(output_dir, f"seed_{seed}")

        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.seed = seed
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

    def _read_existing_data(self) -> Dict:
        """
        Read existing processed BeaverTails data.

        Returns:
            Dictionary with train/test unsafe data
        """
        data = {}

        # Read train unsafe (10k examples)
        train_unsafe_path = self.source_dir / "train" / "unsafe_10k.json"
        if train_unsafe_path.exists():
            with open(train_unsafe_path, "r", encoding="utf-8") as f:
                data["train_unsafe"] = json.load(f)
            print(f"Loaded {len(data['train_unsafe'])} train unsafe examples from {train_unsafe_path}")
        else:
            raise FileNotFoundError(f"Required file not found: {train_unsafe_path}")

        # Read test unsafe (1k examples)
        test_unsafe_path = self.source_dir / "test" / "unsafe_1k.json"
        if test_unsafe_path.exists():
            with open(test_unsafe_path, "r", encoding="utf-8") as f:
                data["test_unsafe"] = json.load(f)
            print(f"Loaded {len(data['test_unsafe'])} test unsafe examples from {test_unsafe_path}")
        else:
            raise FileNotFoundError(f"Required file not found: {test_unsafe_path}")

        return data

    def _extract_user_prompt(self, formatted_prompt: str) -> str:
        """
        Extract the original user prompt from formatted prompt.

        Input:  "Prompt: How do I break into a safe?\\nResponse: "
        Output: "How do I break into a safe?"
        """
        if "Prompt: " in formatted_prompt:
            # Extract content between "Prompt: " and "\nResponse: "
            start = formatted_prompt.index("Prompt: ") + len("Prompt: ")
            end = formatted_prompt.index("\nResponse: ")
            return formatted_prompt[start:end].strip()
        return formatted_prompt.strip()

    def generate_responses_batch(self, prompts: List[str], desc: str = "Generating") -> List[str]:
        responses = []
        
        for i in tqdm(range(0, len(prompts), self.batch_size), desc=desc):
            batch_texts = prompts[i:i + self.batch_size]
            
            # 1. 批量构建 chat template
            batch_messages = [[{"role": "user", "content": p}] for p in batch_texts]
            
            # 2. 批量化 Tokenization
            if "Qwen" in self.model_name:
                inputs = self.tokenizer.apply_chat_template(
                    batch_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    padding=True, # 自动 padding 到当前 batch 最长长度
                    return_dict=True,
                    enable_thinking=False
                ).to(self.device)
            else:
                inputs = self.tokenizer.apply_chat_template(
                    batch_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    padding=True, # 自动 padding 到当前 batch 最长长度
                    return_dict=True
                ).to(self.device)


            # 3. 真正的矩阵并行生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # 4. 批量解码
            input_len = inputs.input_ids.shape[1]
            generated_ids = outputs[:, input_len:]
            batch_responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            responses.extend([r.strip() for r in batch_responses])
            
        return responses
    
    def process(self) -> None:
        """
        Main processing pipeline:
        1. Read existing data
        2. Extract prompts from unsafe examples
        3. Generate safe responses using gemma-2-2b-it
        4. Save new dataset
        """
        print("=" * 60)
        print("BeaverTails_gemma2-2b-it Dataset Generation")
        print("=" * 60)

        # Step 1: Read existing data
        print("\n=== Step 1: Reading existing data ===")
        data = self._read_existing_data()

        # Step 2: Load model
        print("\n=== Step 2: Loading gemma-2-2b-it model ===")
        self._load_model()

        # Step 3: Extract prompts from unsafe examples
        print("\n=== Step 3: Extracting prompts ===")
        train_prompts = [
            self._extract_user_prompt(ex["prompt"])
            for ex in data["train_unsafe"]
        ]
        test_prompts = [
            self._extract_user_prompt(ex["prompt"])
            for ex in data["test_unsafe"]
        ]
        print(f"Extracted {len(train_prompts)} train prompts")
        print(f"Extracted {len(test_prompts)} test prompts")

        # Step 4: Generate safe responses for train data
        print("\n=== Step 4: Generating safe train responses ===")
        print(f"Model: {self.model_name}")
        print(f"Max new tokens: {self.max_new_tokens}")
        print(f"Temperature: {self.temperature}")

        train_safe_responses = self.generate_responses_batch(
            train_prompts,
            desc="Generating train safe responses"
        )

        # Step 5: Generate safe responses for test data
        print("\n=== Step 5: Generating safe test responses ===")
        test_safe_responses = self.generate_responses_batch(
            test_prompts,
            desc="Generating test safe responses"
        )

        # Step 6: Create output data structure
        print("\n=== Step 6: Creating output dataset ===")

        # Create train safe output
        train_safe_output = []
        for original_unsafe, generated_response in zip(data["train_unsafe"], train_safe_responses):
            train_safe_output.append({
                "prompt": original_unsafe["prompt"],  # Keep original formatted prompt
                "completion": generated_response
            })

        # Create test safe output
        test_safe_output = []
        for original_unsafe, generated_response in zip(data["test_unsafe"], test_safe_responses):
            test_safe_output.append({
                "prompt": original_unsafe["prompt"],
                "completion": generated_response
            })

        # Step 7: Save all datasets
        print("\n=== Step 7: Saving datasets ===")

        # Save train/unsafe_10k.json (original unsafe copied)
        train_unsafe_path = self.output_dir / "train" / "unsafe_10k.json"
        with open(train_unsafe_path, "w", encoding="utf-8") as f:
            json.dump(data["train_unsafe"], f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data['train_unsafe'])} examples to {train_unsafe_path}")

        # Save train/safe_10k.json (generated safe)
        train_safe_path = self.output_dir / "train" / "safe_10k.json"
        with open(train_safe_path, "w", encoding="utf-8") as f:
            json.dump(train_safe_output, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(train_safe_output)} examples to {train_safe_path}")

        # Save test/unsafe_1k.json (original unsafe copied)
        test_unsafe_path = self.output_dir / "test" / "unsafe_1k.json"
        with open(test_unsafe_path, "w", encoding="utf-8") as f:
            json.dump(data["test_unsafe"], f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data['test_unsafe'])} examples to {test_unsafe_path}")

        # Save test/safe_1k.json (generated safe)
        test_safe_path = self.output_dir / "test" / "safe_1k.json"
        with open(test_safe_path, "w", encoding="utf-8") as f:
            json.dump(test_safe_output, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(test_safe_output)} examples to {test_safe_path}")

        # Step 8: Save metadata
        print("\n=== Step 8: Saving metadata ===")
        metadata = {
            "dataset_name": "BeaverTails_gemma2-2b-it",
            "source_dataset": f"BeaverTails/seed_{self.seed}",
            "generation_model": self.model_name,
            "generation_params": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "batch_size": self.batch_size
            },
            "statistics": {
                "errors": self.error_count,
                "empty_responses": self.empty_count
            },
            "splits": {
                "train/unsafe_10k.json": {
                    "count": len(data["train_unsafe"]),
                    "description": "Original unsafe responses (copied from source)"
                },
                "train/safe_10k.json": {
                    "count": len(train_safe_output),
                    "description": "Safe responses generated by gemma-2-2b-it"
                },
                "test/unsafe_1k.json": {
                    "count": len(data["test_unsafe"]),
                    "description": "Original unsafe responses (copied from source)"
                },
                "test/safe_1k.json": {
                    "count": len(test_safe_output),
                    "description": "Safe responses generated by gemma-2-2b-it"
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
        print(f"  - train/unsafe_10k.json: {len(data['train_unsafe'])} examples (copied)")
        print(f"  - train/safe_10k.json: {len(train_safe_output)} examples (generated)")
        print(f"  - test/unsafe_1k.json: {len(data['test_unsafe'])} examples (copied)")
        print(f"  - test/safe_1k.json: {len(test_safe_output)} examples (generated)")
        print(f"\nGeneration statistics:")
        print(f"  - Errors: {self.error_count}")
        print(f"  - Empty responses: {self.empty_count}")


if __name__ == "__main__":
    import argparse

    # Import set_seed from existing utils
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from alignment_dynamics.backends.standard.utils import set_seed

    parser = argparse.ArgumentParser(
        description="Create BeaverTails_gemma2-2b-it dataset"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="Source directory with processed BeaverTails data (default: data/processed/safety/BeaverTails/seed_42)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: data/processed/safety/BeaverTails_gemma2-2b-it/seed_42)"
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
        help="Seed for source dataset selection (default: 42)"
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
    generator = InstructDatasetGenerator(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        seed=args.seed,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device
    )

    generator.process()
