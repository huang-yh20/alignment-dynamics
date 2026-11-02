"""
Checkpoint utilities for TRL-based cyclic fine-tuning experiments.

This module provides research-specific checkpoint functionality.
"""

import os
import json
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


def find_closest_to_baseline(
    baseline_score: float,
    stage_dir: str,
    reverse_start_step: int = None,
    recovery_metric: str = "positive_score"
) -> Tuple[str, float, Dict[str, float]]:
    """
    寻找最接近基线但是比基线高的checkpoint

    参数:
        baseline_score: 基线指标分数
        stage_dir: 阶段目录
        reverse_start_step: 可选，仅考虑此值的检查点
        recovery_metric: 用于恢复的指标名称 (默认: "positive_score")

    返回:
        checkpoint_path: 最接近的 checkpoint 路径
        score: 对应的分数
        metrics: 该checkpoint的完整指标
    """
    min_diff = float('inf')
    best_checkpoint = None
    best_metrics = None

    # 遍历所有checkpoint
    for checkpoint_name in os.listdir(stage_dir):
        checkpoint_path = os.path.join(stage_dir, checkpoint_name)
        if not os.path.isdir(checkpoint_path):
            continue

        # 如果指定了reverse_start_step，检查checkpoint名称是否匹配
        if reverse_start_step is not None:
            # 检查checkpoint名称是否以"P{reverse_start_step}-"开头
            expected_prefix = f"P{reverse_start_step}-"
            if not checkpoint_name.startswith(expected_prefix):
                continue

        metadata_path = os.path.join(checkpoint_path, "metadata.json")
        if not os.path.exists(metadata_path):
            continue

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        metrics = metadata.get("metrics", {})
        current_score = metrics.get(recovery_metric, 0)

        # 计算与基线的差值
        diff = abs(current_score - baseline_score)

        # 更新最佳匹配（如果更接近但未跨越基线）
        # 注意：这里使用嵌套if来确保只在同时满足两个条件时才更新
        eps = 0.005
        if diff < min_diff:
            if current_score > baseline_score - eps:
                min_diff = diff
                best_checkpoint = checkpoint_path
                best_metrics = metrics

    # 如果没有找到匹配的checkpoint
    if best_checkpoint is None:
        if reverse_start_step is not None:
            raise ValueError(f"No checkpoints found with reverse_start_step={reverse_start_step} in {stage_dir}")
        else:
            raise ValueError(f"No valid checkpoints found in {stage_dir}")

    return best_checkpoint, best_metrics.get(recovery_metric, 0), best_metrics
