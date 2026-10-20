# -*- coding: utf-8 -*-
"""数据准备:把 kagglehub 下载或手动解压的 12 类原始数据整理为 5 类,按 8:1:1 分层划分。

用法:
    python scripts/prepare_data.py                 # 自动在 data/raw 下寻找数据集目录
    python scripts/prepare_data.py --raw <dir>     # 指定原始数据目录(含 12 个类别子目录)

自动下载(需代理):python scripts/prepare_data.py --download
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, PROCESSED_DIR, RAW_DIR, RAW_TO_TARGET, TEST_SPLIT, VAL_SPLIT, ensure_dirs

SPLIT_RATIOS = {"train": 1 - VAL_SPLIT - TEST_SPLIT, "val": VAL_SPLIT, "test": TEST_SPLIT}


def find_raw_dir() -> Path | None:
    """在 data/raw 下定位包含 garbage-classification 12 类子目录的目录。"""
    candidates = []
    for p in RAW_DIR.rglob("*"):
        if p.is_dir() and any((p / c).is_dir() for c in ("green-glass", "biological")):
            candidates.append(p)
    if not candidates:
        return None
    return min(candidates, key=lambda p: len(p.parts))  # 取最浅的匹配


def download() -> Path:
    import kagglehub  # 需要代理,失败时请改用手动下载
    path = Path(kagglehub.dataset_download("mostafaabla/garbage-classification"))
    print("kagglehub 下载完成:", path)
    return path


def collect_files(raw: Path) -> dict[str, list[Path]]:
    """按目标类别收集图片路径。"""
    files: dict[str, list[Path]] = {c: [] for c in CLASSES}
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for sub in sorted(raw.iterdir()):
        if not sub.is_dir() or sub.name.lower() not in RAW_TO_TARGET:
            continue
        target = RAW_TO_TARGET[sub.name.lower()]
        files[target].extend(f for f in sub.iterdir() if f.suffix.lower() in exts)
    return files


def split_and_copy(files: dict[str, list[Path]]):
    rng = random.Random(42)
    total = 0
    for cls, paths in files.items():
        rng.shuffle(paths)
        n = len(paths)
        n_train = round(n * SPLIT_RATIOS["train"])
        n_val = round(n * SPLIT_RATIOS["val"])
        parts = {
            "train": paths[:n_train],
            "val": paths[n_train:n_train + n_val],
            "test": paths[n_train + n_val:],
        }
        for split, items in parts.items():
            dst = PROCESSED_DIR / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for f in items:
                shutil.copy2(f, dst / f.name)
        total += n
        print(f"{cls:8s}: {n} 张 (train {len(parts['train'])} / val {len(parts['val'])} / test {len(parts['test'])})")
    print(f"合计 {total} 张 → {PROCESSED_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=None, help="原始数据目录(含 12 个类别子目录)")
    ap.add_argument("--download", action="store_true", help="用 kagglehub 自动下载(需代理)")
    args = ap.parse_args()

    ensure_dirs()
    if PROCESSED_DIR.exists() and any(PROCESSED_DIR.iterdir()):
        sys.exit(f"{PROCESSED_DIR} 已有内容,如需重新整理请先删除该目录。")

    raw = args.raw or (download() if args.download else None) or find_raw_dir()
    if raw is None:
        sys.exit(
            "未找到原始数据。两种获取方式:\n"
            "1) 自动:python scripts/prepare_data.py --download  (需代理)\n"
            "2) 手动:从 https://www.kaggle.com/datasets/mostafaabla/garbage-classification "
            "下载 zip 解压到 data/raw/ 后重跑本脚本"
        )
    print("原始数据目录:", raw)
    files = collect_files(raw)
    if not any(files.values()):
        sys.exit("在原始目录下未匹配到任何 12 类子目录,请检查目录结构。")
    split_and_copy(files)


if __name__ == "__main__":
    main()
