"""Aggregate all model results on test set and generate comparison charts (binary)
Usage: python scripts/05_evaluate_all.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.config import RESULTS_PATH, FIGURE_DIR
from src.plot import plot_model_comparison, plot_per_class_f1


def main():
    if not RESULTS_PATH.exists():
        print(f"{RESULTS_PATH} not found. Please run training scripts first.")
        return

    df = pd.read_csv(RESULTS_PATH)
    print(f"\n{'='*60}")
    print("Model Results Summary (sorted by Macro-F1)")
    print(f"{'='*60}")
    df_sorted = df.sort_values("macro_f1", ascending=False)
    cols = ["model", "accuracy", "macro_f1", "f1_差评", "f1_好评"]
    print(df_sorted[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"{'='*60}")

    best = df_sorted.iloc[0]
    print(f"\nBest model: {best['model']} (Macro-F1: {best['macro_f1']:.4f})")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print("\nGenerating comparison charts...")
    plot_model_comparison(RESULTS_PATH, str(FIGURE_DIR / "model_comparison.png"))
    print(f"  -> {FIGURE_DIR / 'model_comparison.png'}")
    plot_per_class_f1(RESULTS_PATH, str(FIGURE_DIR / "per_class_f1.png"))
    print(f"  -> {FIGURE_DIR / 'per_class_f1.png'}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
