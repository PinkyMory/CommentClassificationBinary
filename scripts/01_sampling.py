"""Merge multiple review datasets, balance pos/neg, then stratified split 8:1:1.

Supports two input formats:
  - Star rating: columns like 评论内容 + 评分 (1-5), converted via star_to_label
  - Pre-labeled: columns like review + label (0/1), used directly

Default: merges 训练集.csv and online_shopping_10_cats.csv together.
Usage:  python scripts/01_sampling.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import SEED, PROCESSED_DIR, TRAIN_PATH, VAL_PATH, TEST_PATH, star_to_label

np.random.seed(SEED)

TEXT_CANDIDATES = ["review", "评论内容", "comment", "text", "content", "评论标题"]
STAR_CANDIDATES = ["评分", "star", "score", "rating", "stars", "星级"]
LABEL_CANDIDATES = ["label"]


def detect_format(df: pd.DataFrame) -> dict:
    """Detect text column and data format (star-rating or pre-labeled)."""
    text_col = None
    for candidate in TEXT_CANDIDATES:
        for col in df.columns:
            if candidate == col.lower() or candidate in col:
                text_col = col
                break
        if text_col:
            break

    if text_col is None:
        raise RuntimeError(f"Cannot detect text column. Available: {list(df.columns)}")

    star_col = None
    for candidate in STAR_CANDIDATES:
        for col in df.columns:
            if col != text_col and (candidate == col.lower() or candidate in col):
                star_col = col
                break
        if star_col:
            break

    label_col = None
    for candidate in LABEL_CANDIDATES:
        for col in df.columns:
            if col != text_col and candidate == col.lower():
                label_col = col
                break
        if label_col:
            break

    if star_col:
        fmt = "star"
        result = {"text_col": text_col, "star_col": star_col, "format": fmt}
    elif label_col is not None:
        fmt = "label"
        result = {"text_col": text_col, "label_col": label_col, "format": fmt}
    else:
        raise RuntimeError(
            f"Cannot detect rating or label column. Available: {list(df.columns)}"
        )

    print(f"  {Path(df.attrs.get('path', '?')).name}: "
          f"fmt={fmt}, text='{text_col}'"
          f"{', rating=' + repr(star_col) if star_col else ''}"
          f"{', label=' + repr(label_col) if label_col else ''}")
    return result


def load_and_normalize(input_path: str) -> pd.DataFrame:
    """Load one file, detect format, normalize to [text, label]."""
    path = Path(input_path)
    print(f"Loading: {path}")
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif path.suffix == ".json":
        df = pd.read_json(path)
    elif path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    df.attrs["path"] = str(path)
    info = detect_format(df)
    text_col = info["text_col"]

    if info["format"] == "star":
        df["label"] = df[info["star_col"]].apply(star_to_label)
    else:
        df["label"] = pd.to_numeric(df[info["label_col"]], errors="coerce")

    # Clean
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].astype(str).str.strip() != ""]
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    # Drop any non-0/1 labels (e.g. neutral=2 in some datasets)
    df = df[df["label"].isin([0, 1])]

    df["text"] = df[text_col].astype(str).str.strip()
    return df[["text", "label"]].reset_index(drop=True)


def show_distribution(df: pd.DataFrame, title: str = ""):
    """Print label distribution with a bar chart."""
    label_names = {0: "差评", 1: "好评"}
    total = len(df)
    if title:
        print(f"\n{'='*50}")
        print(f"  {title}: {total:,} rows")
        print(f"{'='*50}")
    else:
        print(f"\nTotal: {total:,} rows")

    label_dist = df["label"].value_counts().sort_index()
    for lbl, count in label_dist.items():
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"  {label_names[lbl]}: {count:>8,} ({pct:5.1f}%) {bar}")

    if len(label_dist) == 2:
        max_c = label_dist.max()
        min_c = label_dist.min()
        print(f"  比例 (max/min): {max_c / min_c:.1f}:1")


def balance(df: pd.DataFrame) -> pd.DataFrame:
    """Downsample majority class to match minority count."""
    neg = df[df["label"] == 0]
    pos = df[df["label"] == 1]
    n_neg, n_pos = len(neg), len(pos)

    if n_neg == n_pos:
        print("\n  数据已均衡，无需下采样。")
        return df

    target = min(n_neg, n_pos)
    majority_name = "好评" if n_pos > n_neg else "差评"
    minority_name = "差评" if n_pos > n_neg else "好评"

    if n_neg < n_pos:
        keep = neg
        downsample = pos.sample(n=target, random_state=SEED)
    else:
        keep = pos
        downsample = neg.sample(n=target, random_state=SEED)

    print(f"\n  下采样 {majority_name}: {max(n_neg, n_pos):,} -> {target:,}")
    print(f"  保留全部 {minority_name}: {min(n_neg, n_pos):,}")
    print(f"  均衡后总量: {target * 2:,}")

    return pd.concat([keep, downsample], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


def split_and_save(df: pd.DataFrame):
    """Stratified 8:1:1 split and save to processed/."""
    train, temp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=SEED)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False, encoding="utf-8-sig")
    val.to_csv(VAL_PATH, index=False, encoding="utf-8-sig")
    test.to_csv(TEST_PATH, index=False, encoding="utf-8-sig")

    label_names = {0: "差评", 1: "好评"}
    print(f"\n{'='*50}")
    print(f"  划分结果 (8:1:1)")
    print(f"{'='*50}")
    for name, path in [("Train", TRAIN_PATH), ("Val", VAL_PATH), ("Test", TEST_PATH)]:
        sub = pd.read_csv(path)
        dist = sub["label"].value_counts().sort_index()
        parts = ", ".join(f"{label_names[k]}: {v:,}" for k, v in dist.items())
        print(f"  {name}: {len(sub):>8,} rows ({parts})")


def main():
    default_inputs = [
        "data/raw/训练集.csv",
        "data/raw/online_shopping_10_cats.csv",
    ]
    parser = argparse.ArgumentParser(
        description="Merge review datasets, balance pos/neg, stratified split 8:1:1"
    )
    parser.add_argument(
        "--input", type=str, nargs="+", default=default_inputs,
        help="Input CSV files (default: 训练集.csv + online_shopping_10_cats.csv)"
    )
    parser.add_argument(
        "--no-balance", action="store_true",
        help="Skip class balancing"
    )
    args = parser.parse_args()

    # Load and normalize all inputs
    dfs = []
    for p in args.input:
        dfs.append(load_and_normalize(p))

    merged = pd.concat(dfs, ignore_index=True)
    show_distribution(merged, "合并后（原始）")

    if not args.no_balance:
        merged = balance(merged)
        show_distribution(merged, "均衡后")

    split_and_save(merged)
    print("\n完成！")


if __name__ == "__main__":
    main()
