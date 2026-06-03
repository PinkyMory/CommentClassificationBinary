"""Stratified sampling: JD review data -> binary dataset (exclude neutral), train/val/test = 8:1:1
Usage: python scripts/01_sampling.py
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


def load_raw_data(input_path: str) -> pd.DataFrame:
    print(f"Loading raw data: {input_path}")
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".tsv"):
        df = pd.read_csv(input_path, sep="\t")
    elif input_path.endswith(".json"):
        df = pd.read_json(input_path)
    elif input_path.endswith(".jsonl"):
        df = pd.read_json(input_path, lines=True)
    else:
        raise ValueError(f"Unsupported format: {input_path}")
    return df


def find_text_and_star_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Auto-detect comment column and rating column"""
    text_candidates = ["评论内容", "评论标题", "content", "comment", "text", "review", "内容"]
    star_candidates = ["评分", "star", "score", "rating", "stars", "星级"]

    text_col = None
    star_col = None

    for candidate in text_candidates:
        for col in df.columns:
            if candidate in col.lower():
                text_col = col
                break
        if text_col:
            break

    for candidate in star_candidates:
        for col in df.columns:
            if candidate in col.lower():
                star_col = col
                break
        if star_col:
            break

    if text_col is None or star_col is None:
        print(f"Available columns: {list(df.columns)}")
        raise RuntimeError("Could not auto-detect comment or rating column. Use --text-col and --star-col.")
    print(f"Detected -> comment: '{text_col}', rating: '{star_col}'")
    return text_col, star_col


def prepare_data(df: pd.DataFrame, text_col: str, star_col: str) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df[star_col].apply(star_to_label)
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].astype(str).str.strip() != ""]
    col_map = {text_col: "text"}
    if star_col != "star":
        col_map[star_col] = "star"
    df = df.rename(columns=col_map)
    cols_to_keep = ["text", "label"]
    if "star" in df.columns:
        cols_to_keep = ["text", "star", "label"]
    return df[cols_to_keep].reset_index(drop=True)


def show_distributions(df: pd.DataFrame):
    label_names = {0: "差评", 1: "好评"}
    print(f"\nTotal valid data (binary, star<=3 as negative): {len(df):,} rows\n")
    if "star" in df.columns:
        print("Star distribution:")
        star_dist = df["star"].value_counts().sort_index()
        for star, count in star_dist.items():
            print(f"  {int(star)} star: {count:>6,} ({100*count/len(df):5.1f}%)")
    print("\n2-class label distribution:")
    label_dist = df["label"].value_counts().sort_index()
    for label, count in label_dist.items():
        pct = 100 * count / len(df)
        bar = "█" * int(pct / 2)
        print(f"  {label_names[label]}: {count:>6,} ({pct:5.1f}%) {bar}")
    if len(label_dist) == 2:
        max_count = label_dist.max()
        min_count = label_dist.min()
        print(f"\nImbalance ratio (max/min): {max_count / min_count:.1f}:1")


def split_and_save(df: pd.DataFrame):
    train, temp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=SEED)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False, encoding="utf-8-sig")
    val.to_csv(VAL_PATH, index=False, encoding="utf-8-sig")
    test.to_csv(TEST_PATH, index=False, encoding="utf-8-sig")

    label_names = {0: "差评", 1: "好评"}
    print(f"\nDatasets saved (stratified 8:1:1 split):")
    for name, path in [("Train", TRAIN_PATH), ("Val", VAL_PATH), ("Test", TEST_PATH)]:
        sub = pd.read_csv(path)
        dist = sub["label"].value_counts().sort_index()
        parts = ", ".join(f"{label_names[k]}: {v:,}" for k, v in dist.items())
        print(f"  {name}: {len(sub):,} rows -> {path}")
        print(f"         {parts}")


def main():
    parser = argparse.ArgumentParser(description="JD review data stratified sampling (binary, star<=3 as negative)")
    parser.add_argument("--input", type=str, default="data/raw/训练集.csv")
    parser.add_argument("--text-col", type=str, default=None)
    parser.add_argument("--star-col", type=str, default=None)
    args = parser.parse_args()

    df = load_raw_data(args.input)
    if args.text_col and args.star_col:
        text_col, star_col = args.text_col, args.star_col
    else:
        text_col, star_col = find_text_and_star_columns(df)

    df = prepare_data(df, text_col, star_col)
    show_distributions(df)
    split_and_save(df)
    print("\nSampling complete!")


if __name__ == "__main__":
    main()
