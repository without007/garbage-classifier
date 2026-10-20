# -*- coding: utf-8 -*-
"""训练入口。

用法:
    python -m src.train --model cnn        # 自建轻量 CNN(基础组)
    python -m src.train --model resnet18   # ResNet-18 迁移学习(对比组)
"""
import argparse
import json
import time

import matplotlib
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import CKPT_DIR, EPOCHS, FIG_DIR, LOG_DIR, LR, LR_HEAD, NUM_CLASSES, ensure_dirs
from .data import make_loaders
from .models import build_model

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # 中文标签
plt.rcParams["axes.unicode_minus"] = False


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "resnet18"], default="cnn")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--scheduler", choices=["step", "cosine"], default="cosine")
    ap.add_argument("--weighted", action="store_true", help="按训练集频率对类别加权,缓解类别不平衡")
    ap.add_argument("--unfreeze", action="store_true", help="解冻 ResNet-18 的 layer4 参与微调")
    ap.add_argument("--tag", default="", help="实验标签,追加到产物文件名,避免覆盖已有结果")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " ⚠ 无可用 GPU,训练会较慢"))

    train_loader, val_loader, _ = make_loaders()
    model = build_model(args.model, unfreeze=args.unfreeze).to(device)

    # 迁移学习:分类头用 LR_HEAD;解冻的 layer4 用更小学习率,避免破坏预训练特征
    if args.model == "resnet18":
        head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("fc.")]
        backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc.")]
        groups = [{"params": head_params, "lr": LR_HEAD}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": LR_HEAD / 10})
            print(f"解冻 layer4:{sum(p.numel() for p in backbone_params) / 1e6:.2f}M 参数,"
                  f"学习率 {LR_HEAD / 10:g}")
        optimizer = Adam(groups, weight_decay=1e-4)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = Adam([{"params": params, "lr": LR}], weight_decay=1e-4)
    scheduler = (StepLR(optimizer, step_size=5, gamma=0.5) if args.scheduler == "step"
                 else CosineAnnealingLR(optimizer, T_max=args.epochs))
    criterion = nn.CrossEntropyLoss()
    if args.weighted:
        import numpy as np
        freq = np.bincount([y for _, y in train_loader.dataset.samples], minlength=NUM_CLASSES)
        # w_i = N / (C * n_i),使各类权重均值为 1
        w = torch.tensor(len(train_loader.dataset) / (NUM_CLASSES * freq), dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=w)
        print("类别权重:", dict(zip(train_loader.dataset.classes, [round(x, 3) for x in w.tolist()])))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_path = 0.0, CKPT_DIR / f"best_{args.model}{suffix}.pth"
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        marker = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"model": args.model, "classes": train_loader.dataset.classes,
                        "state_dict": model.state_dict(), "epoch": epoch}, best_path)
            marker = "  ← 已保存最优权重"
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f}{marker}")

    log_path = LOG_DIR / f"train_{args.model}{suffix}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps(
        {"args": vars(args), "best_val_acc": best_val_acc, "minutes": round((time.time() - start) / 60, 1),
         "history": history}, ensure_ascii=False, indent=2), encoding="utf-8")

    # 收敛曲线
    epochs = range(1, args.epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title(f"{args.model} Loss 收敛曲线")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title(f"{args.model} Accuracy 曲线")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig_path = FIG_DIR / f"curves_{args.model}{suffix}.png"
    fig.savefig(fig_path, dpi=150)
    print(f"完成:最优 val acc {best_val_acc:.4f},权重 {best_path},曲线 {fig_path},日志 {log_path}")


if __name__ == "__main__":
    main()
