import random
import os
import torch
import numpy as np
import yaml
import json
from typing import Dict, Any, Optional, List
import logging


def set_seed(seed: int):
    """设置所有随机种子"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """加载 YAML 配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_device(device: str = "auto") -> str:
    """获取训练设备"""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def compute_l2_distance(model1, model2) -> float:
    """计算模型参数间的 L2 距离"""
    distance = 0.0
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            raise ValueError(f"Parameter name mismatch: {name1} vs {name2}")
        distance += torch.norm(param1 - param2).item() ** 2
    return distance ** 0.5


def save_json(data: Dict[str, Any], path: str):
    """保存数据为 JSON 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    """从 JSON 文件加载数据"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


