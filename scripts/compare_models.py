# -*- coding: utf-8 -*-
"""对比实验汇总:读取两个模型的训练日志与评估结果,输出对比表与柱状图。

用法: python scripts/compare_models.py
产物: outputs/figures/compare_models.png, outputs/logs/compare.json
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import CLASS_NAMES_CN, FIG_DIR, LOG_DIR, ensure_dirs

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

MODEL_LABELS = {
    "cnn": "自建轻量 CNN",
    "resnet18": "ResNet-18(冻结)",
    "resnet18_unfrozen": "ResNet-18(解冻 layer4)",
}


def discover_variants() -> list[str]:
    """从 eval_<variant>.json 文件名发现已评估的实验变体。"""
    return [p.stem[len("eval_"):] for p in sorted(LOG_DIR.glob("eval_*.json"))]


def latest_train_log(variant: str) -> dict | None:
    """按精确变体名匹配训练日志(避免 resnet18 误匹配 resnet18_unfrozen)。"""
    pat = re.compile(rf"^train_{re.escape(variant)}_\d{{8}}_\d{{6}}\.json$")
    logs = sorted(p for p in LOG_DIR.glob("train_*.json") if pat.match(p.name))
    if not logs:
        return None
    d = json.loads(logs[-1].read_text(encoding="utf-8"))
    d["_file"] = logs[-1].name
    return d


def parse_eval(variant: str) -> dict | None:
    p = LOG_DIR / f"eval_{variant}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def parse_report(text: str) -> dict:
    """从 classification_report 文本里取出 overall accuracy 与 macro/weighted avg。"""
    out = {}
    m = re.search(r"accuracy\s+([\d.]+)\s+(\d+)", text)
    if m:
        out["accuracy"] = float(m.group(1))
    for name in ("macro avg", "weighted avg"):
        m = re.search(rf"{name}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", text)
        if m:
            out[name] = {"precision": float(m.group(1)), "recall": float(m.group(2)),
                         "f1": float(m.group(3))}
    return out


def parse_per_class(text: str) -> dict:
    """逐类别取 precision/recall/f1。"""
    rows = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] not in ("accuracy",) and not parts[0].startswith("macro"):
            try:
                rows[parts[0]] = {"precision": float(parts[1]), "recall": float(parts[2]), "f1": float(parts[3])}
            except ValueError:
                continue
    return rows


def main():
    ensure_dirs()
    summary = {}
    for variant in discover_variants():
        tr, ev = latest_train_log(variant), parse_eval(variant)
        if not tr or not ev:
            print(f"跳过 {variant}:缺少训练或评估日志")
            continue
        metrics = parse_report(ev["report"])
        summary[variant] = {
            "best_val_acc": tr["best_val_acc"],
            "minutes": tr["minutes"],
            "epochs": len(tr["history"]["train_acc"]),
            "test_accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro avg", {}).get("f1"),
            "per_class": parse_per_class(ev["report"]),
            "wrong": len(ev["confused"]),
        }

    if not summary:
        sys.exit("没有可对比的模型日志,请先训练并评估。")

    print(f"\n{'模型':22s} {'最优val':>8s} {'测试acc':>8s} {'macro-F1':>9s} {'训练时长':>9s} {'错分':>5s}")
    for m, s in summary.items():
        print(f"{MODEL_LABELS.get(m, m):22s} {s['best_val_acc']:8.4f} {s['test_accuracy'] or 0:8.4f} "
              f"{s['macro_f1'] or 0:9.4f} {s['minutes']:7.1f}分 {s['wrong']:5d}")

    # 逐类别 F1 对比
    classes = [CLASS_NAMES_CN[c] for c in ("glass", "kitchen", "metal", "paper", "plastic")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    keys = ["best_val_acc", "test_accuracy", "macro_f1"]
    names = ["最优验证准确率", "测试集准确率", "Macro-F1"]
    width = 0.8 / max(len(summary), 1)
    for i, (m, s) in enumerate(summary.items()):
        vals = [s[k] or 0 for k in keys]
        ax.bar([x + i * width for x in range(len(keys))], vals, width, label=MODEL_LABELS.get(m, m))
        for x, v in zip(range(len(keys)), vals):
            ax.text(x + i * width, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks([x + width * (len(summary) - 1) / 2 for x in range(len(keys))])
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_title("整体指标对比")
    ax.legend()

    ax = axes[1]
    for i, (m, s) in enumerate(summary.items()):
        vals = [s["per_class"].get(c, {}).get("f1", 0) for c in classes]
        ax.bar([x + i * width for x in range(len(classes))], vals, width, label=MODEL_LABELS.get(m, m))
    ax.set_xticks([x + width * (len(summary) - 1) / 2 for x in range(len(classes))])
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.05)
    ax.set_title("各类别 F1 对比")
    ax.legend()

    fig.tight_layout()
    fig_path = FIG_DIR / "compare_models.png"
    fig.savefig(fig_path, dpi=150)
    (LOG_DIR / "compare.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n对比图: {fig_path}\n汇总: {LOG_DIR / 'compare.json'}")


if __name__ == "__main__":
    main()
