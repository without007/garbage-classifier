# -*- coding: utf-8 -*-
"""全局配置:路径、类别映射、超参数。"""
from pathlib import Path

# 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"            # 原始下载(kaggle zip / kagglehub 缓存)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"  # 整理后的 train/val/test
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

# 目标类别(与设计方案 4.1 节一致)
CLASSES = ["plastic", "metal", "glass", "paper", "kitchen"]
CLASS_NAMES_CN = {
    "plastic": "塑料",
    "metal": "金属/易拉罐",
    "glass": "玻璃",
    "paper": "纸张/纸盒",
    "kitchen": "厨余",
}
NUM_CLASSES = len(CLASSES)

# Kaggle 数据集 mostafaabla/garbage-classification 的 12 个原始目录名 → 目标类别
RAW_TO_TARGET = {
    "plastic": "plastic",
    "metal": "metal",
    "green-glass": "glass",
    "brown-glass": "glass",
    "white-glass": "glass",
    "paper": "paper",
    "cardboard": "paper",
    "biological": "kitchen",
    # trash / battery / clothes / shoes 等不纳入训练
}

# 训练超参数
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
LR_HEAD = 3e-4          # 迁移学习分类头学习率
WEIGHT_DECAY = 1e-4
DROPOUT = 0.5
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
SEED = 42
NUM_WORKERS = 2

# ImageNet 归一化(两个模型统一)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def ensure_dirs():
    for d in (RAW_DIR, PROCESSED_DIR, CKPT_DIR, FIG_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
