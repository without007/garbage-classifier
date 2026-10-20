# -*- coding: utf-8 -*-
"""模型定义:自建轻量级 3 层 CNN 与 ResNet-18 迁移学习。"""
import torch.nn as nn
from torchvision import models

from .config import DROPOUT, NUM_CLASSES


class SimpleCNN(nn.Module):
    """三层特征提取(Conv-BN-ReLU-MaxPool) + 全连接分类头。"""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = DROPOUT):
        super().__init__()
        ch = [3, 64, 128, 256]
        layers = []
        for i in range(3):
            layers += [
                nn.Conv2d(ch[i], ch[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(ch[i + 1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*layers)  # 224 -> 28
        self.classifier = nn.Sequential(
            # 全局平均池化把 256×28×28 压成 256 维,避免巨型全连接层
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_resnet18(pretrained: bool = True, unfreeze: bool = False) -> nn.Module:
    """加载 ImageNet 预训练 ResNet-18,冻结特征层并替换分类头。

    unfreeze=True 时同时解冻最后一个残差块 layer4 参与微调。
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    if unfreeze:
        for p in model.layer4.parameters():
            p.requires_grad = True
    return model


def build_model(name: str, unfreeze: bool = False) -> nn.Module:
    if name == "cnn":
        return SimpleCNN()
    if name == "resnet18":
        return build_resnet18(unfreeze=unfreeze)
    raise ValueError(f"未知模型: {name}(可选 cnn / resnet18)")
