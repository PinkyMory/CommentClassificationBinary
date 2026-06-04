"""
模型评估汇总脚本（Step 5）
=============================================================================
读取 outputs/results.csv 中所有模型的测试集结果，进行汇总展示和可视化。

功能：
  1. 按 Macro-F1 降序排列展示所有模型的核心指标
  2. 标注最佳模型
  3. 生成两个对比图表：
     - model_comparison.png:  所有模型的 Accuracy/Macro-F1/Weighted-F1 柱状图
     - per_class_f1.png:      所有模型在差评/好评上的 F1 对比

前置条件：
  需要至少运行过一个训练脚本（02/03/04），results.csv 中存在数据。

使用方法：
  python scripts/05_evaluate_all.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.config import RESULTS_PATH, FIGURE_DIR
from src.plot import plot_model_comparison, plot_per_class_f1


def main():
    """Main entry: read results.csv, print sorted summary, generate charts."""
    if not RESULTS_PATH.exists():
        print(f"{RESULTS_PATH} not found. Please run training scripts first.")
        print("（未找到 results.csv，请先运行训练脚本 02/03/04）")
        return

    # ---- 读取并排序 ----
    df = pd.read_csv(RESULTS_PATH)

    print(f"\n{'='*60}")
    print("Model Results Summary (sorted by Macro-F1)")
    print(f"{'='*60}")

    # 按 Macro-F1 降序排列
    df_sorted = df.sort_values("macro_f1", ascending=False)

    # 展示核心指标
    cols = ["model", "accuracy", "macro_f1", "f1_差评", "f1_好评"]
    print(df_sorted[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"{'='*60}")

    # ---- 最佳模型 ----
    best = df_sorted.iloc[0]
    print(f"\nBest model: {best['model']} (Macro-F1: {best['macro_f1']:.4f})")
    print(f"  → 建议在 demo 中选择此模型获得最佳预测效果")

    # ---- 生成对比图表 ----
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print("\nGenerating comparison charts...")

    # 整体指标对比（Accuracy / Macro-F1 / Weighted-F1）
    plot_model_comparison(RESULTS_PATH, str(FIGURE_DIR / "model_comparison.png"))
    print(f"  -> {FIGURE_DIR / 'model_comparison.png'}")

    # 每类 F1 对比（差评 / 好评）
    # 这张图能暴露模型是否偏向某一类
    plot_per_class_f1(RESULTS_PATH, str(FIGURE_DIR / "per_class_f1.png"))
    print(f"  -> {FIGURE_DIR / 'per_class_f1.png'}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
