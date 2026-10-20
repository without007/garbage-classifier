# -*- coding: utf-8 -*-
"""测试集评估:Accuracy/Precision/Recall/F1 + 混淆矩阵热力图 + 易混淆样本清单。

用法: python -m src.evaluate --model cnn
"""
import argparse
import json

import matplotlib
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import CKPT_DIR, CLASSES, CLASS_NAMES_CN, FIG_DIR, PROCESSED_DIR, ensure_dirs
from .data import EVAL_TRANSFORM, make_loaders
from .models import build_model

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "resnet18"], default="cnn")
    ap.add_argument("--tag", default="", help="实验标签,对应 train 时使用的 --tag")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT_DIR / f"best_{args.model}{suffix}.pth", map_location=device)
    classes = ckpt["classes"]
    model = build_model(args.model).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_ds = datasets.ImageFolder(PROCESSED_DIR / "test", transform=EVAL_TRANSFORM)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

    all_preds, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    labels_cn = [CLASS_NAMES_CN.get(c, c) for c in classes]
    report = classification_report(y_true, y_pred, target_names=labels_cn, digits=4)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_cn, yticklabels=labels_cn, ax=ax)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"{args.model} 混淆矩阵(测试集)")
    fig.tight_layout()
    cm_path = FIG_DIR / f"confusion_{args.model}{suffix}.png"
    fig.savefig(cm_path, dpi=150)

    # 易混淆样本:被错分的图片清单(供误差分析)
    idxs = np.where(y_pred != y_true)[0]
    confusion_log = [
        {"path": str(test_ds.samples[i][0]), "true": classes[int(y_true[i])], "pred": classes[int(y_pred[i])]}
        for i in idxs
    ]
    out = FIG_DIR.parent / "logs" / f"eval_{args.model}{suffix}.json"
    out.write_text(json.dumps({"report": report, "confused": confusion_log[:200]},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"混淆矩阵: {cm_path} | 评估明细: {out} | 错分样本 {len(confusion_log)} 条")


if __name__ == "__main__":
    main()
