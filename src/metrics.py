"""
模型评估指标模块
=============================================================================
提供统一的评估接口，所有训练脚本和评估脚本共用。

主要功能：
  - compute_metrics():      计算完整的分类评估指标集
  - print_metrics():        格式化打印评估结果
  - append_to_results_csv(): 将模型结果追加/更新到 results.csv
  - save_metrics_to_file(): 将评估报告保存为文本文件

results.csv 的作用：
  汇总所有模型在测试集上的表现，供 05_evaluate_all.py 生成对比图表，
  以及 app/demo.py 前端展示模型下拉框的指标数据。
  每行一个模型，列包括：model, accuracy, macro_f1, f1_差评, f1_好评, ...
"""

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
import pandas as pd
import numpy as np


def compute_metrics(y_true, y_pred, label_names=None) -> dict:
    """计算二分类的完整评估指标集

    返回字典包含：
      - accuracy:           整体准确率
      - macro_f1:           宏平均 F1（简单算术平均，对类别不均衡更敏感）
      - weighted_f1:        加权平均 F1（按样本数加权，与整体分布一致）
      - precision_per_class: 每个类别的精确率 [neg_precision, pos_precision]
      - recall_per_class:   每个类别的召回率 [neg_recall, pos_recall]
      - f1_per_class:       每个类别的 F1 [neg_f1, pos_f1]
      - confusion_matrix:   2×2 混淆矩阵 [[TN, FP], [FN, TP]]
      - classification_report: sklearn 的分类报告字符串

    以 macro-F1 为主要评估指标：因为数据经过均衡处理后两类样本数相同，
    macro-F1 公平地衡量模型在两个类别上的表现。

    Args:
        y_true:      真实标签列表
        y_pred:      预测标签列表
        label_names: 类别名称列表（默认 ["差评", "好评"]）

    Returns:
        包含所有指标的字典
    """
    if label_names is None:
        label_names = ["差评", "好评"]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "precision_per_class": precision_score(y_true, y_pred, average=None).tolist(),
        "recall_per_class": recall_score(y_true, y_pred, average=None).tolist(),
        "f1_per_class": f1_score(y_true, y_pred, average=None).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=label_names, digits=4
        ),
    }


def print_metrics(metrics: dict):
    """格式化打印评估结果到控制台

    输出内容包括：
      - 整体指标：Accuracy, Macro-F1, Weighted-F1
      - 每类指标：F1, Precision, Recall（差评 / 好评）
      - 混淆矩阵：表格形式（True·差评 / True·好评 vs Pred→差评 / Pred→好评）
      - sklearn 分类报告：含 per-class 的 precision/recall/f1/support

    Args:
        metrics: compute_metrics() 返回的指标字典
    """
    print(f"\n{'='*50}")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:    {metrics['weighted_f1']:.4f}")
    print(f"\nPer-class F1:")
    for name, f1, p, r in zip(
        ["差评", "好评"],
        metrics["f1_per_class"],
        metrics["precision_per_class"],
        metrics["recall_per_class"],
    ):
        print(f"  {name}:  F1={f1:.4f}  Precision={p:.4f}  Recall={r:.4f}")
    print(f"\nConfusion Matrix:")
    cm = np.array(metrics["confusion_matrix"])
    print(f"          Pred→差评 好评")
    for i, name in enumerate(["True·差评", "True·好评"]):
        print(f"  {name}:  {cm[i][0]:4d}  {cm[i][1]:4d}")
    print(f"\n{metrics['classification_report']}")
    print(f"{'='*50}\n")


def append_to_results_csv(results_csv_path: str, model_name: str, metrics: dict):
    """将模型评估结果追加或更新到 results.csv

    行为：
      - 如果 results.csv 已存在：替换同名模型的行，保留其他模型
      - 如果不存在：新建文件
      - 这样同一模型重复运行只会保留最新结果

    保存的列：
      model, accuracy, macro_f1, weighted_f1,
      f1_差评, f1_好评, precision_差评, precision_好评,
      recall_差评, recall_好评

    Args:
        results_csv_path: results.csv 文件路径
        model_name:       模型名称（如 "TextCNN", "roberta"）
        metrics:          compute_metrics() 返回的指标字典
    """
    import os
    # 构建新行，提取关键指标
    row = {
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "f1_差评": metrics["f1_per_class"][0],
        "f1_好评": metrics["f1_per_class"][1],
        "precision_差评": metrics["precision_per_class"][0],
        "precision_好评": metrics["precision_per_class"][1],
        "recall_差评": metrics["recall_per_class"][0],
        "recall_好评": metrics["recall_per_class"][1],
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(results_csv_path):
        # 已存在：移除同名旧行，追加新行（实现"更新"效果）
        df_existing = pd.read_csv(results_csv_path)
        df_existing = df_existing[df_existing["model"] != model_name]
        df_out = pd.concat([df_existing, df_row], ignore_index=True)
    else:
        df_out = df_row
    df_out.to_csv(results_csv_path, index=False)
    print(f"Results written to {results_csv_path}")


def save_metrics_to_file(metrics: dict, save_path: str, model_name: str = ""):
    """将评估指标保存为格式化的文本报告文件

    输出格式与控制台打印一致，便于离线查看和存档。
    文件编码为 UTF-8，确保中文正常显示。

    Args:
        metrics:    compute_metrics() 返回的指标字典
        save_path:  保存路径（.txt 文件）
        model_name: 模型名称，写入报告标题
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy:     {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1:     {metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1:  {metrics['weighted_f1']:.4f}\n")
        f.write("\nPer-class:\n")
        labels = ["差评", "好评"]
        for i, name in enumerate(labels):
            f.write(f"  {name}:  F1={metrics['f1_per_class'][i]:.4f}  "
                    f"Precision={metrics['precision_per_class'][i]:.4f}  "
                    f"Recall={metrics['recall_per_class'][i]:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        cm = metrics["confusion_matrix"]
        f.write(f"          Pred→差评  好评\n")
        for i, name in enumerate(["True·差评", "True·好评"]):
            f.write(f"  {name}:  {cm[i][0]:4d}  {cm[i][1]:4d}\n")
        f.write("\n" + metrics["classification_report"] + "\n")
    print(f"Evaluation report saved to {save_path}")
