"""
Training callbacks for TRL-based cyclic fine-tuning experiments.

This module provides custom callbacks for:
- Specific step saving
- Periodic evaluation
- Checkpoint metadata management
- Recovery search (stage 3)
"""

import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class SpecificStepSaveCallback(TrainerCallback):
    """按特定步数列表保存模型的回调

    支持非均匀间隔的保存点，例如 [128, 256, 512, 1024] 或 [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    与 TRL 的 save_steps（整数间隔）不同，此回调精确控制保存时机。
    """

    def __init__(self, save_steps_list: List[int]):
        """
        参数:
            save_steps_list: 要保存模型的步数列表，例如 [128, 256, 512, 1024]
        """
        self.save_steps_list = set(save_steps_list)  # 使用 set 提高查找效率

    def on_step_end(self, args, state, control, **kwargs):
        """
        在每步结束时检查是否应该保存
        如果当前步数在列表中，触发保存
        """
        if state.global_step in self.save_steps_list:
            control.should_save = True
            if state.is_world_process_zero:
                logger.info(f"Triggering save at step {state.global_step} (specific step save)")
        return control


class EvaluationCallback(TrainerCallback):
    """定期评估回调 - 使用项目现有的Evaluator"""

    def __init__(self, evaluator, eval_every: int, stage: str):
        self.evaluator = evaluator
        self.eval_every = eval_every
        self.stage = stage

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """每N步评估一次，只在主进程执行"""
        # All ranks participate in the evaluator collective operations.
        if state.global_step % self.eval_every == 0 and state.global_step > 0:
            logger.info(f"Running evaluation at step {state.global_step}")
            self.evaluator.set_stage_info(step=state.global_step)
            metrics = self.evaluator.evaluate(model=model)

            # 更新Trainer的log_history（TRL自动处理）
            control.should_log = True
        return control


class MetadataCallback(TrainerCallback):
    """为TRL保存的checkpoint添加自定义元数据和命名，并在保存前计算评估指标"""

    def __init__(self, evaluator, stage: str, save_path_prefix: str = "P",
                 save_steps_list: List[int] = None):
        """
        参数:
            evaluator: Evaluator实例，用于计算评估指标
            stage: 当前训练阶段
            save_path_prefix: checkpoint名称前缀
            save_steps_list: 保存步数列表，用于判断何时需要预先计算指标
        """
        self.evaluator = evaluator
        self.stage = stage
        self.save_path_prefix = save_path_prefix
        self.save_steps_list = set(save_steps_list or [])
        self._cached_metrics = {}  # 缓存当前步的评估指标

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        在每步结束时检查是否即将保存checkpoint
        如果是，提前计算评估指标并缓存
        """
        # 检查当前步是否在保存列表中
        if state.global_step in self.save_steps_list:
            logger.info(f"Pre-computing metrics for checkpoint save at step {state.global_step}")

            # 设置评估器状态
            self.evaluator.set_stage_info(step=state.global_step, stage=self.stage)

            # 计算评估指标
            metrics = self.evaluator.evaluate(model=model)
            self._cached_metrics[state.global_step] = metrics

            logger.info(f"Metrics pre-computed for step {state.global_step}: {metrics}")

    def on_save(self, args, state, control, **kwargs):
        """
        TRL保存checkpoint**之后**重命名并添加元数据（包含预先计算的评估指标）
        只在主进程处理检查点重命名和元数据

        执行顺序:
        1. SpecificStepSaveCallback.on_step_end() - 检查是否在保存列表中，设置 should_save=True
        2. MetadataCallback.on_step_end() - 提前计算并缓存指标
        3. TRL保存checkpoint
        4. MetadataCallback.on_save() - 重命名并添加metadata（包含缓存的指标）
        """
        # 只在主进程处理检查点重命名和元数据
        if not state.is_world_process_zero:
            return

        # TRL保存为 checkpoint-{step}，我们需要重命名为 P{step}
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        if os.path.exists(checkpoint_dir):
            # 重命名
            new_name = f"{self.save_path_prefix}{state.global_step}"
            new_dir = os.path.join(args.output_dir, new_name)
            os.rename(checkpoint_dir, new_dir)

            # 获取缓存的评估指标
            metrics = self._cached_metrics.get(state.global_step, {})

            # 添加自定义元数据（TRL不保存这些）
            metadata = {
                "stage": self.stage,
                "step": state.global_step,
                "metrics": metrics,  # 保存评估指标到metadata
                "timestamp": datetime.now().isoformat()
            }
            with open(os.path.join(new_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Checkpoint saved at step {state.global_step} with metrics: {metrics}")

            # 清理缓存（可选，避免内存占用）
            if state.global_step in self._cached_metrics:
                del self._cached_metrics[state.global_step]


class RecoverySearchCallback(TrainerCallback):
    """Stage 3恢复搜索回调 - 实时评估并搜索最佳匹配"""

    def __init__(
        self,
        baseline_score: float,
        recovery_metric: str,
        threshold: float,
        output_dir: str,
        reverse_start_step: int,
        evaluator
    ):
        self.baseline_score = baseline_score
        self.recovery_metric = recovery_metric
        self.threshold = threshold
        self.output_dir = output_dir
        self.reverse_start_step = reverse_start_step
        self.evaluator = evaluator
        self.best_diff = float('inf')
        self.best_match_path = None
        self.best_match_step = None
        self.best_match_metrics = None

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """每步评估并搜索最佳匹配，只在主进程执行"""
        # 只在主进程执行恢复搜索

        # 评估当前checkpoint
        self.evaluator.set_stage_info(step=state.global_step, stage="stage_3_recovery")
        metrics = self.evaluator.evaluate(model=model)
        current_score = metrics.get(self.recovery_metric, 0)
        current_diff = abs(current_score - self.baseline_score)

        logger.info(f"Step {state.global_step}: {self.recovery_metric}={current_score:.4f}, diff={current_diff:.4f}")

        if current_diff < self.threshold or current_score < self.baseline_score:
            self.best_diff = current_diff
            self.best_match_step = state.global_step
            self.best_match_metrics = metrics

            logger.info(f"Recovery complete at step {state.global_step}, diff={current_diff:.4f}")
            control.should_save = True
            control.should_training_stop = True

        return control

    def on_save(self, args, state, control, **kwargs):
        """
        TRL保存checkpoint**之后**重命名并添加元数据
        只在主进程处理检查点重命名和元数据

        TRL保存为 checkpoint-{step}，重命名为 best_match_P{reverse_start_step}_N{step}
        并添加包含恢复搜索特定字段的元数据
        """
        # 只在主进程处理检查点重命名和元数据
        if not state.is_world_process_zero:
            return

        # TRL保存为 checkpoint-{step}，我们需要重命名为 best_match_P{reverse_start_step}_N{step}
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        if os.path.exists(checkpoint_dir):
            # 使用最佳匹配的步数进行重命名
            new_name = f"best_match_P{self.reverse_start_step}_N{self.best_match_step}"
            new_dir = os.path.join(args.output_dir, new_name)
            os.rename(checkpoint_dir, new_dir)
            self.best_match_path = new_dir

            # 添加自定义元数据（包含恢复搜索特定字段）
            metadata = {
                "stage": "stage_3_recovery",
                "step": self.best_match_step,
                "metrics": self.best_match_metrics,
                "baseline_score": self.baseline_score,
                "diff": self.best_diff,
                "timestamp": datetime.now().isoformat()
            }
            with open(os.path.join(new_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved best match checkpoint from step {self.best_match_step} with diff={self.best_diff:.4f}")
