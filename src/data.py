# -*- coding: utf-8 -*-
"""数据变换与 DataLoader。"""
import torch
from torchvision import datasets, transforms

from .config import BATCH_SIZE, CLASSES, IMAGENET_MEAN, IMAGENET_STD, NUM_WORKERS, PROCESSED_DIR, SEED, IMG_SIZE

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def make_loaders():
    """从 data/processed/{train,val,test} 构建 DataLoader。"""
    common = dict(transform=None)
    train_ds = datasets.ImageFolder(PROCESSED_DIR / "train", transform=TRAIN_TRANSFORM)
    val_ds = datasets.ImageFolder(PROCESSED_DIR / "val", transform=EVAL_TRANSFORM)
    test_ds = datasets.ImageFolder(PROCESSED_DIR / "test", transform=EVAL_TRANSFORM)

    # ImageFolder 的类别顺序按目录名排序,需与 config.CLASSES 对齐
    assert train_ds.classes == sorted(CLASSES), f"数据目录类别 {train_ds.classes} 与配置不符"

    g = torch.Generator().manual_seed(SEED)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, generator=g)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader, test_loader
