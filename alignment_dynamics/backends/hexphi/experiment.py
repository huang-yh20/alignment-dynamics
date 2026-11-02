"""
Cyclic fine-tuning experiment using TRL SFTTrainer.

This module implements a 4-stage cyclic fine-tuning experiment:
- Stage 0: Baseline evaluation
- Stage 1: Forward fine-tuning (positive sentiment)
- Stage 2: Reverse fine-tuning (negative sentiment)
- Stage 3: Recovery search (find baseline-equivalent checkpoint)
- Stage 4: Forward fine-tuning again (consistency verification)
"""

import os
import argparse
import logging
import json
import torch
import gc
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator

from .callbacks import (
    SpecificStepSaveCallback,
    EvaluationCallback,
    MetadataCallback,
    RecoverySearchCallback
)
from .checkpoint_utils import find_closest_to_baseline
from .evaluator import Evaluator
from .utils import load_config, set_seed, save_json

logger = logging.getLogger(__name__)


def clear_gpu_memory():
    """清理GPU显存的辅助函数"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class CyclicExperiment:
    """循环微调实验 - TRL简化版"""

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.results_dir = None
        self.experiment_name = None

        # 初始化 Accelerator
        self.accelerator = Accelerator()

        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def setup_experiment(self, seed: int, output_dir: str = None):
        """设置实验环境"""
        set_seed(seed)
        self.seed = seed
        self.experiment_name = f"alignment_dynamics_{seed}"
        self.results_dir = output_dir or os.path.join("results", self.experiment_name)
        os.makedirs(self.results_dir, exist_ok=True)

        # 保存实验配置
        config_with_seed = self.config.copy()
        config_with_seed["seed"] = seed
        if self.accelerator.is_main_process:
            save_json(config_with_seed, os.path.join(self.results_dir, "config.json"))

        self.logger.info(f"Experiment initialized: {self.experiment_name}")
        self.logger.info(f"Results directory: {self.results_dir}")

    def _setup_model_and_tokenizer(self, checkpoint_path: str = None):
        """通用模型设置 - 仅加载模型权重，不恢复训练状态

        参数:
            checkpoint_path: checkpoint路径（如果为None则加载预训练模型）

        注意:
            - 此方法仅加载模型权重，不恢复训练状态（optimizer、scheduler等）
            - 对于stage_3的resume，使用TRL的resume_from_checkpoint参数
        """
        # 加载tokenizer
        if checkpoint_path:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.config["model"]["name"])

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 加载模型权重（不恢复训练状态）
        # DeepSpeed 会自动处理设备分配，无需指定 device_map
        if checkpoint_path:
            # 直接加载完整模型
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
            )
        else:
            # 加载预训练模型
            model = AutoModelForCausalLM.from_pretrained(
                self.config["model"]["name"],
                torch_dtype=torch.bfloat16,
            )

        return model, tokenizer

    def stage_0_baseline(self):
        """Stage 0: 基线评估"""
        self.logger.info("Starting Stage 0: Baseline evaluation")

        # 加载模型
        model, tokenizer = self._setup_model_and_tokenizer()
        model = model.to(self.accelerator.device)

        # 初始化评估器
        compute_metrics = self.config.get("evaluation", {}).get("compute", None)
        evaluator = Evaluator(
            tokenizer=tokenizer,
            sentiment_model=self.config["evaluation"].get("sentiment_model", None),
            safety_model=self.config["evaluation"].get("safety_model", None),
            device=self.config["model"]["device"],
            max_length=self.config["model"]["max_length"],
            compute_metrics=compute_metrics,
            logs_dir=os.path.join(self.results_dir, "logs"),
            stage="stage_0_baseline",
            step=0,
            accelerator=self.accelerator
        )

        # 加载测试数据
        test_positive_path = os.path.join(
            self.config["data"]["processed_dir"],
            f"seed_{self.config['data']['seed']}",
            self.config["data"]["test_positive_data"]
        )
        test_negative_path = os.path.join(
            self.config["data"]["processed_dir"],
            f"seed_{self.config['data']['seed']}",
            self.config["data"]["test_negative_data"]
        )
        evaluator.load_test_data(test_positive_path, test_negative_path)

        # 评估
        baseline_metrics = evaluator.evaluate(model=model)

        # 保存checkpoint（手动，因为不需要训练）
        # 只在主进程保存文件
        if self.accelerator.is_main_process:
            stage_dir = os.path.join(self.results_dir, "stage_0_baseline")
            os.makedirs(stage_dir, exist_ok=True)
            checkpoint_dir = os.path.join(stage_dir, "P0")

            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

            # 保存元数据（包含完整的评估指标）
            metadata = {
                "stage": "stage_0_baseline",
                "step": 0,
                "metrics": baseline_metrics,  # 保存完整的评估指标
                "timestamp": datetime.now().isoformat()
            }
            save_json(metadata, os.path.join(checkpoint_dir, "metadata.json"))

            # 保存指标
            save_json(baseline_metrics, os.path.join(self.results_dir, "stage_0_baseline", "metrics.json"))

            self.logger.info(f"Baseline metrics: {baseline_metrics}")

        clear_gpu_memory()
        return baseline_metrics

    def stage_1_forward(self):
        """Run forward supervised fine-tuning."""
        return self._stage1_sft()


    def _stage1_sft(self):
        """Stage 1: 正向微调 - 使用TRL SFTTrainer

        模型加载策略:
        - 直接加载预训练模型（不加载stage_0 checkpoint的训练状态）
        - Stage 0只是评估，不需要恢复其训练状态
        """
        self.logger.info("Starting Stage 1: Forward fine-tuning (SFT)")

        # 加载预训练模型（不加载任何checkpoint的训练状态）
        model, tokenizer = self._setup_model_and_tokenizer(checkpoint_path=None)

        # 初始化评估器
        compute_metrics = self.config.get("evaluation", {}).get("compute", None)
        evaluator = Evaluator(
            tokenizer=tokenizer,
            sentiment_model=self.config["evaluation"].get("sentiment_model", None),
            safety_model=self.config["evaluation"].get("safety_model", None),
            device=self.config["model"]["device"],
            max_length=self.config["model"]["max_length"],
            compute_metrics=compute_metrics,
            logs_dir=os.path.join(self.results_dir, "logs"),
            stage="stage_1_forward",
            accelerator=self.accelerator
        )
        test_positive_path = os.path.join(
            self.config["data"]["processed_dir"],
            f"seed_{self.config['data']['seed']}",
            self.config["data"]["test_positive_data"]
        )
        test_negative_path = os.path.join(
            self.config["data"]["processed_dir"],
            f"seed_{self.config['data']['seed']}",
            self.config["data"]["test_negative_data"]
        )
        evaluator.load_test_data(test_positive_path, test_negative_path)

        # 加载训练数据 - **直接使用TRL支持的格式！**
        positive_data_path = os.path.join(
            self.config["data"]["processed_dir"],
            f"seed_{self.config['data']['seed']}",
            self.config["data"]["train_positive_data"]
        )
        train_dataset = load_dataset(
            "json",
            data_files=positive_data_path,
            split="train"
        )

        # 设置输出目录
        stage_dir = os.path.join(self.results_dir, "stage_1_forward")

        # TRL SFTConfig - 禁用自动保存，完全由 Callback 控制
        max_step = max(self.config["saving"]["stage_1_save_steps"])
        sft_config = SFTConfig(
            seed=self.seed,
            data_seed=self.seed,
            output_dir=stage_dir,
            learning_rate=float(self.config["training"]["learning_rate"]),
            lr_scheduler_type="constant",  # 使用constant学习率调度（与原始代码一致）
            per_device_train_batch_size=self.config["training"]["per_device_train_batch_size"],
            max_grad_norm=self.config["training"]["max_grad_norm"],
            max_steps=max_step,
            logging_steps=10,
            save_strategy="steps",
            save_steps=10**9,   # 极大值，永远不会自然触发
            save_total_limit=None,  # 不限制保存数量
            save_only_model=True,  # 只保存模型权重，不保存训练状态
            report_to=[],
            gradient_checkpointing=True,
            max_length=self.config["model"]["max_length"],
            completion_only_loss=True,  # 只在completion上计算loss
            # DeepSpeed ZeRO-2 + BF16 支持
            deepspeed=self.config.get("deepspeed", "deepspeed_config_zero2.json"),
            bf16=True,
            ddp_find_unused_parameters=False,
            gradient_accumulation_steps=self.config.get("training", {}).get("gradient_accumulation_steps", 1),
        )

        # 创建回调
        callbacks = [
            SpecificStepSaveCallback(
                save_steps_list=self.config["saving"]["stage_1_save_steps"]  # [128, 256, 512, 1024]
            ),
            EvaluationCallback(
                evaluator=evaluator,
                eval_every=self.config["evaluation"]["eval_every"],
                stage="stage_1"
            ),
            MetadataCallback(
                evaluator=evaluator,  # 传入evaluator以计算评估指标
                stage="stage_1_forward",
                save_path_prefix="P",
                save_steps_list=self.config["saving"]["stage_1_save_steps"]  # 传入保存步数列表
            )
        ]

        # 创建SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            callbacks=callbacks
        )

        # 训练 - TRL自动处理checkpoint保存和加载
        trainer.train()

        self.logger.info(f"Stage 1 completed")

        clear_gpu_memory()
        return {}


    def stage_2_reverse(self):
        """Stage 2: 反向微调

        模型加载策略:
        - 从stage_1的P{reverse_start_step}加载模型权重（不加载训练状态）
        - 全量微调：直接加载模型checkpoint
        - 数据：使用负样本数据
        - 命名：save_path_prefix = "P{reverse_start_step}-N"
        """
        self.logger.info("Starting Stage 2: Reverse fine-tuning")

        # 加载 stage_1 的特定 checkpoint（仅模型权重，不恢复训练状态）
        reverse_start_step = self.config["saving"]["reverse_start_step"]
        from_checkpoint_path = os.path.join(
            self.results_dir,
            "stage_1_forward",
            f"P{reverse_start_step}"  
        )

        # 加载模型权重（不恢复训练状态）
        model, tokenizer = self._setup_model_and_tokenizer(
            checkpoint_path=from_checkpoint_path
        )

        # 初始化评估器
        compute_metrics = self.config.get("evaluation", {}).get("compute", None)
        evaluator = Evaluator(
            tokenizer=tokenizer,
            sentiment_model=self.config["evaluation"].get("sentiment_model", None),
            safety_model=self.config["evaluation"].get("safety_model", None),
            device=self.config["model"]["device"],
            max_length=self.config["model"]["max_length"],
            compute_metrics=compute_metrics,
            logs_dir=os.path.join(self.results_dir, "logs"),
            stage="stage_2_reverse",
            accelerator=self.accelerator
        )
        test_positive_path = os.path.join(
            self.config["data"]["processed_dir"],
            f"seed_{self.config['data']['seed']}",
            self.config["data"]["test_positive_data"]
        )
        test_negative_path = os.path.join(
            self.config["data"]["processed_dir"],
            f"seed_{self.config['data']['seed']}",
            self.config["data"]["test_negative_data"]
        )
        evaluator.load_test_data(test_positive_path, test_negative_path)

        # 加载训练数据（负样本）
        negative_data_path = os.path.join(
            self.config["data"]["processed_dir"],
            f"seed_{self.config['data']['seed']}",
            self.config["data"]["train_negative_data"]
        )
        train_dataset = load_dataset(
            "json",
            data_files=negative_data_path,
            split="train"
        )

        # 设置输出目录
        stage_dir = os.path.join(self.results_dir, "stage_2_reverse")

        # TRL SFTConfig
        max_step = max(self.config["saving"]["stage_2_save_steps"])
        sft_config = SFTConfig(
            seed=self.seed,
            data_seed=self.seed,
            output_dir=stage_dir,
            learning_rate=float(self.config["training"]["learning_rate"]),
            lr_scheduler_type="constant",  # 使用constant学习率调度（与原始代码一致）
            per_device_train_batch_size=self.config["training"]["per_device_train_batch_size"],
            max_grad_norm=self.config["training"]["max_grad_norm"],
            max_steps=max_step,
            logging_steps=10,
            save_strategy="steps",
            save_steps=10**9,   # 极大值，永远不会自然触发
            save_only_model=True,   # 对于HEx-PHI实验只保存模型
            save_total_limit=None,
            report_to=[],
            gradient_checkpointing=True,
            max_length=self.config["model"]["max_length"],
            completion_only_loss=True,
            # DeepSpeed ZeRO-2 + BF16 支持
            deepspeed=self.config.get("deepspeed", "deepspeed_config_zero2.json"),
            bf16=True,
            ddp_find_unused_parameters=False,
            gradient_accumulation_steps=self.config.get("training", {}).get("gradient_accumulation_steps", 1),
        )

        # 创建回调
        callbacks = [
            SpecificStepSaveCallback(
                save_steps_list=self.config["saving"]["stage_2_save_steps"]  
            ),
            EvaluationCallback(
                evaluator=evaluator,
                eval_every=self.config["evaluation"]["eval_every"],
                stage="stage_2"
            ),
            MetadataCallback(
                evaluator=evaluator,
                stage="stage_2_reverse",
                save_path_prefix=f"P{reverse_start_step}-N",  # 带前缀的命名
                save_steps_list=self.config["saving"]["stage_2_save_steps"]
            )
        ]

        # 创建SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            callbacks=callbacks
        )

        # 训练
        trainer.train()

        self.logger.info(f"Stage 2 completed")

        clear_gpu_memory()
        return {}

    def stage_3_recovery(self, baseline_metrics):
        """Stage 3: 恢复搜索 - 使用RecoverySearchCallback

        模型加载策略:
        - 使用TRL的resume_from_checkpoint参数恢复完整训练状态
        - 包括：模型权重、optimizer状态、scheduler状态、训练步数等
        - 从stage_2选定的checkpoint继续训练
        """
        self.logger.info("Starting Stage 3: Recovery")

        # 找到 stage_2 中最接近基线的 checkpoint 作为起点
        stage_2_dir = os.path.join(self.results_dir, "stage_2_reverse")
        reverse_start_step = self.config["saving"]["reverse_start_step"]
        recovery_metric = self.config["evaluation"]["recovery_metric"]

        # 使用 find_closest_to_baseline 找到最佳起始 checkpoint
        # 该函数会搜索所有 P{reverse_start_step}-N* 开头的 checkpoint
        # 找到 metrics 中 recovery_metric 最接近 baseline 但仍然高于 baseline 的那个
        baseline_score = baseline_metrics[recovery_metric]
        closest_checkpoint, closest_score, metrics = find_closest_to_baseline(
            baseline_score=baseline_score,
            stage_dir=stage_2_dir,
            reverse_start_step=reverse_start_step,
            recovery_metric=recovery_metric
        )

        self.logger.info(f"Stage 3 resuming from checkpoint: {closest_checkpoint} "
                        f"(score: {closest_score:.4f}, baseline: {baseline_score:.4f})")

        # 注意：不需要手动加载模型和tokenizer
        # TRL的resume_from_checkpoint会自动处理模型、optimizer、scheduler等所有状态的恢复

        # 设置输出目录
        stage_dir = os.path.join(self.results_dir, "stage_3_recovery")

        # 加载训练数据（反向数据）- **直接使用TRL支持的格式！**
        negative_data_path = os.path.join(
            self.config["data"]["processed_dir"],
            f"seed_{self.config['data']['seed']}",
            self.config["data"]["train_negative_data"]
        )
        train_dataset = load_dataset(
            "json",
            data_files=negative_data_path,
            split="train"
        )

        # TRL SFTConfig
        # 读取 checkpoint 的 metadata.json 获取步数
        metadata_path = os.path.join(closest_checkpoint, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                checkpoint_step = metadata.get("step", 0)
        else:
            # 如果没有 metadata.json，使用 0 作为默认值（不应发生）
            checkpoint_step = 0
            self.logger.warning(f"No metadata.json found in {closest_checkpoint}, assuming step 0")

        max_search_steps = self.config["saving"]["stage_3_max_search_steps"]
        max_steps = checkpoint_step + max_search_steps

        self.logger.info(f"Stage 3: checkpoint_step={checkpoint_step}, "
                         f"max_search_steps={max_search_steps}, "
                         f"max_steps={max_steps}")

        sft_config = SFTConfig(
            seed=self.seed,
            data_seed=self.seed,
            output_dir=stage_dir,
            learning_rate=float(self.config["training"]["learning_rate"]),
            lr_scheduler_type="constant",  # 使用constant学习率调度（与原始代码一致）
            per_device_train_batch_size=self.config["training"]["per_device_train_batch_size"],
            max_grad_norm=self.config["training"]["max_grad_norm"],
            max_steps=max_steps,
            logging_steps=1,
            save_strategy="steps",
            save_steps=10**9,   # 极大值，永远不会自然触发
            save_only_model=True,  # 只保存模型权重，不保存训练状态
            report_to=[],
            gradient_checkpointing=True,
            max_length=self.config["model"]["max_length"],
            completion_only_loss=True,
            # DeepSpeed ZeRO-2 + BF16 支持
            deepspeed=self.config.get("deepspeed", "deepspeed_config_zero2.json"),
            bf16=True,
            ddp_find_unused_parameters=False,
            gradient_accumulation_steps=self.config.get("training", {}).get("gradient_accumulation_steps", 1),
        )

        # 注意：创建SFTTrainer时不传入model参数
        # TRL会根据resume_from_checkpoint自动加载模型和训练状态

        # 创建RecoverySearchCallback（evaluator稍后设置）
        recovery_callback = RecoverySearchCallback(
            baseline_score=baseline_metrics[self.config["evaluation"]["recovery_metric"]],
            recovery_metric=self.config["evaluation"]["recovery_metric"],
            threshold=self.config["evaluation"]["recovery_threshold"],
            output_dir=os.path.join(stage_dir),
            reverse_start_step=reverse_start_step,
            evaluator=None  # 稍后设置
        )

        # 仅用于加载模型结构
        model, tokenizer = self._setup_model_and_tokenizer(checkpoint_path=closest_checkpoint)

        # 创建SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            callbacks=[recovery_callback]
        )

        # 现在可以初始化评估器（因为trainer.model已经可用）
        compute_metrics = self.config.get("evaluation", {}).get("compute", None)
        evaluator = Evaluator(
            tokenizer=tokenizer,
            sentiment_model=self.config["evaluation"].get("sentiment_model", None),
            safety_model=self.config["evaluation"].get("safety_model", None),
            device=self.config["model"]["device"],
            max_length=self.config["model"]["max_length"],
            compute_metrics=compute_metrics,
            logs_dir=os.path.join(self.results_dir, "logs"),
            stage="stage_3_recovery",
            accelerator=self.accelerator
        )
        test_positive_path = os.path.join(
            self.config["data"]["processed_dir"],
            f"seed_{self.config['data']['seed']}",
            self.config["data"]["test_positive_data"]
        )
        test_negative_path = os.path.join(
            self.config["data"]["processed_dir"],
            f"seed_{self.config['data']['seed']}",
            self.config["data"]["test_negative_data"]
        )
        evaluator.load_test_data(test_positive_path, test_negative_path)

        # 设置evaluator到callback
        recovery_callback.evaluator = evaluator

        # 训练直到RecoverySearchCallback停止
        trainer.train(resume_from_checkpoint=closest_checkpoint)

        self.logger.info(f"Stage 3 completed")

        clear_gpu_memory()

        # 返回最佳匹配checkpoint路径
        self.accelerator.wait_for_everyone()
        return self._load_recovery_path()

    def stage_4_forward_again(self, recovery_path):
        """Stage 4: 再次正向微调

        模型加载策略（与stage_2相同）:
        - 从stage_3的recovery_path加载模型权重（不加载训练状态）
        - 全量微调：直接加载模型checkpoint
        - 数据：使用正向样本数据
        - 命名：save_path_prefix = f"P{reverse_start_step}-N{stage3_recovery_steps}-P"
          - reverse_start_step: 从config获取，表示从stage_2的哪个checkpoint开始
          - stage3_recovery_steps: 从stage_3 recovery checkpoint的metadata中提取
          - 最终格式例如: P256-N42-P{step}
        - 保存步数：stage_1_save_steps: [128, 256, 512, 1024]
        """
        self.logger.info("Starting Stage 4: Forward fine-tuning again")

        # 获取 reverse_start_step 用于命名
        reverse_start_step = self.config["saving"]["reverse_start_step"]

        # 从stage_3 recovery checkpoint的metadata中提取步数
        metadata_path = os.path.join(recovery_path, "metadata.json")
        with open(metadata_path, 'r') as f:
            recovery_metadata = json.load(f)
        stage3_recovery_steps = recovery_metadata["step"]

        # 加载模型权重（不恢复训练状态）
        model, tokenizer = self._setup_model_and_tokenizer(
            checkpoint_path=recovery_path
        )

        # 初始化评估器
        compute_metrics = self.config.get("evaluation", {}).get("compute", None)
        evaluator = Evaluator(
            tokenizer=tokenizer,
            sentiment_model=self.config["evaluation"].get("sentiment_model", None),
            safety_model=self.config["evaluation"].get("safety_model", None),
            device=self.config["model"]["device"],
            max_length=self.config["model"]["max_length"],
            compute_metrics=compute_metrics,
            logs_dir=os.path.join(self.results_dir, "logs"),
            stage="stage_4_forward",
            accelerator=self.accelerator
        )
        test_positive_path = os.path.join(
            self.config["data"]["processed_dir"],
            f"seed_{self.config['data']['seed']}",
            self.config["data"]["test_positive_data"]
        )
        test_negative_path = os.path.join(
            self.config["data"]["processed_dir"],
            f"seed_{self.config['data']['seed']}",
            self.config["data"]["test_negative_data"]
        )
        evaluator.load_test_data(test_positive_path, test_negative_path)

        # 加载训练数据（正向样本）
        positive_data_path = os.path.join(
            self.config["data"]["processed_dir"],
            f"seed_{self.config['data']['seed']}",
            self.config["data"]["train_positive_data"]
        )
        train_dataset = load_dataset(
            "json",
            data_files=positive_data_path,
            split="train"
        )

        # 设置输出目录
        stage_dir = os.path.join(self.results_dir, "stage_4_forward")

        # TRL SFTConfig
        max_step = max(self.config["saving"]["stage_1_save_steps"])
        sft_config = SFTConfig(
            seed=self.seed,
            data_seed=self.seed,
            output_dir=stage_dir,
            learning_rate=float(self.config["training"]["learning_rate"]),
            lr_scheduler_type="constant",  # 使用constant学习率调度（与原始代码一致）
            per_device_train_batch_size=self.config["training"]["per_device_train_batch_size"],
            max_grad_norm=self.config["training"]["max_grad_norm"],
            max_steps=max_step,
            logging_steps=10,
            save_strategy="steps",
            save_steps=10**9,   # 极大值，永远不会自然触发
            save_total_limit=None,
            save_only_model=True,  # 只保存模型权重，不保存训练状态
            report_to=[],
            gradient_checkpointing=True,
            max_length=self.config["model"]["max_length"],
            completion_only_loss=True,
            # DeepSpeed ZeRO-2 + BF16 支持
            deepspeed=self.config.get("deepspeed", "deepspeed_config_zero2.json"),
            bf16=True,
            ddp_find_unused_parameters=False,
            gradient_accumulation_steps=self.config.get("training", {}).get("gradient_accumulation_steps", 1),
        )

        # 创建回调
        callbacks = [
            SpecificStepSaveCallback(
                save_steps_list=self.config["saving"]["stage_1_save_steps"]  # [128, 256, 512, 1024]
            ),
            EvaluationCallback(
                evaluator=evaluator,
                eval_every=self.config["evaluation"]["eval_every"],
                stage="stage_4"
            ),
            MetadataCallback(
                evaluator=evaluator,
                stage="stage_4_forward",
                save_path_prefix=f"P{reverse_start_step}-N{stage3_recovery_steps}-P",  # 动态前缀
                save_steps_list=self.config["saving"]["stage_1_save_steps"]
            )
        ]

        # 创建SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            callbacks=callbacks
        )

        # 训练
        trainer.train()

        self.logger.info(f"Stage 4 completed")

        clear_gpu_memory()
        return {}

    def run(self, seed: int, output_dir: str = None, stage: int = None):
        """运行实验"""
        self.setup_experiment(seed, output_dir)

        if stage is not None:
            if stage == 0:
                return self.stage_0_baseline()
            elif stage == 1:
                return self.stage_1_forward()
            elif stage == 2:
                return self.stage_2_reverse()
            elif stage == 3:
                baseline_metrics = self._load_baseline_metrics()
                return self.stage_3_recovery(baseline_metrics)
            elif stage == 4:
                recovery_path = self._load_recovery_path()
                return self.stage_4_forward_again(recovery_path)
        else:
            # 完整实验
            baseline_metrics = self.stage_0_baseline()
            stage_1_metrics = self.stage_1_forward()
            stage_2_metrics = self.stage_2_reverse()
            recovery_path = self.stage_3_recovery(baseline_metrics)
            stage_4_metrics = self.stage_4_forward_again(recovery_path)

            # 生成报告（只在主进程保存）
            report = {
                "experiment_name": self.experiment_name,
                "seed": seed,
                "results": {
                    "stage_0": baseline_metrics,
                    "stage_1": stage_1_metrics,
                    "stage_2": stage_2_metrics,
                    "stage_3": {"recovery_checkpoint": recovery_path},
                    "stage_4": stage_4_metrics
                }
            }
            if self.accelerator.is_main_process:
                save_json(report, os.path.join(self.results_dir, "experiment_report.json"))
            return report

    def _load_baseline_metrics(self):
        """加载stage 0指标"""
        metrics_path = os.path.join(self.results_dir, "stage_0_baseline", "metrics.json")
        with open(metrics_path, 'r') as f:
            return json.load(f)

    def _load_recovery_path(self):
        """加载stage 3恢复checkpoint路径"""
        import glob
        stage_3_checkpoints_dir = os.path.join(self.results_dir, "stage_3_recovery")
        reverse_start_step = self.config["saving"]["reverse_start_step"]
        pattern = os.path.join(stage_3_checkpoints_dir, f"best_match_P{reverse_start_step}_N*")
        matches = glob.glob(pattern)

        if not matches:
            # Also check for the case where stage 3 didn't find a better match
            closest_match_path = os.path.join(self.results_dir, "stage_3_recovery", "best_match")
            if os.path.exists(closest_match_path):
                return closest_match_path
            raise FileNotFoundError("Stage 3 recovery checkpoint not found. Run stage 3 first.")

        if len(matches) > 1:
            raise ValueError("Multiple recovery checkpoints found; use a fresh run directory.")

        return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--stage", type=int, default=None, choices=[0, 1, 2, 3, 4], help="Run single stage (0-4), default runs all stages")
    args = parser.parse_args()

    # 运行实验
    experiment = CyclicExperiment(args.config)
    experiment.run(args.seed, args.output_dir, args.stage)


if __name__ == "__main__":
    main()
