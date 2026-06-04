"""
可视化绘图模块
=============================================================================
基于 matplotlib + seaborn，为模型评估和对比提供标准化的图表。

所有绘图函数使用 "Agg" 后端（无头服务器兼容），不依赖 GUI。
中文字体优先使用 SimHei（黑体），回退到 DejaVu Sans。

主要功能：
  - plot_confusion_matrix():  混淆矩阵热力图（单个模型）
  - plot_training_curves():   训练过程的 Loss 和 F1 曲线（DL 模型）
  - plot_model_comparison():  所有模型的指标对比柱状图（Accuracy/Macro-F1/Weighted-F1）
  - plot_per_class_f1():      所有模型的每类 F1 对比柱状图（差评/好评）
"""

import matplotlib
# 使用 Agg 后端，支持无 GUI 的服务器环境运行
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 设置中文字体，确保图表中的中文标签正常渲染
# SimHei 是 Windows 系统自带的黑体，Linux 服务器可能需要安装中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
# 解决负号 "-" 显示为方块的问题
plt.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(
    cm: np.ndarray, labels: list[str], save_path: str = None,
    title: str = "Confusion Matrix"
):
    """绘制单个模型的混淆矩阵热力图

    使用 seaborn heatmap 渲染，颜色深浅表示数量。
    x 轴 = 预测标签，y 轴 = 真实标签。
    每个格子内标注具体的样本数量。

    Args:
        cm:        2×2 混淆矩阵 [[TN, FP], [FN, TP]]
        labels:    类别名称列表，如 ["差评", "好评"]
        save_path: 保存路径，None 则只显示不保存
        title:     图表标题
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_curves(history: dict, save_path: str = None):
    """绘制深度学习模型的训练过程曲线

    包含两个子图：
      左图：训练 Loss 和 验证 Loss 随 epoch 的变化
      右图：验证集 Macro-F1 随 epoch 的变化

    用于判断：
      - 是否过拟合（train_loss 持续下降但 val_loss 上升）
      - 是否需要更多 epoch（val_f1 仍在上升趋势中）
      - EarlyStopping 是否在合适的位置停止

    Args:
        history:   训练历史字典，需包含：
                   - "train_loss": 每 epoch 的训练损失列表
                   - "val_loss":   每 epoch 的验证损失列表
                   - "val_f1":     每 epoch 的验证 Macro-F1 列表
        save_path: 保存路径，None 则只显示不保存
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    # 左图：Loss 曲线
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()

    # 右图：验证集 Macro-F1 曲线
    ax2.plot(epochs, history["val_f1"], "g-", label="Val Macro-F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro-F1")
    ax2.set_title("Val Macro-F1")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_model_comparison(results_csv_path: str, save_path: str = None):
    """绘制所有模型的整体指标对比柱状图

    从 results.csv 读取所有模型结果，绘制三个指标的分组柱状图：
      - Accuracy（蓝色）
      - Macro-F1（橙色）
      - Weighted-F1（绿色）

    每个模型一组三柱，便于横向对比不同模型的整体表现。

    Args:
        results_csv_path: results.csv 文件路径
        save_path:        图表保存路径
    """
    df = pd.read_csv(results_csv_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.2  # 每组三根柱的宽度

    ax.bar(x - width, df["accuracy"], width, label="Accuracy")
    ax.bar(x, df["macro_f1"], width, label="Macro-F1")
    ax.bar(x + width, df["weighted_f1"], width, label="Weighted-F1")

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20)  # 旋转 20° 避免标签重叠
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (Binary)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)  # 将 y 轴固定到 0-1.05，便于公平对比
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_per_class_f1(results_csv_path: str, save_path: str = None):
    """绘制所有模型在各类别上的 F1 对比柱状图

    从 results.csv 读取数据，绘制每个模型在两个类别上的 F1：
      - 差评 F1（蓝色）
      - 好评 F1（橙色）

    这个图能暴露模型的类别偏向问题：
      如果某模型的好评 F1 很高但差评 F1 很低，说明模型偏向多数类，
      可能在差评识别上能力不足。

    Args:
        results_csv_path: results.csv 文件路径
        save_path:        图表保存路径
    """
    df = pd.read_csv(results_csv_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.3

    ax.bar(x - width/2, df["f1_差评"], width, label="Negative F1")
    ax.bar(x + width/2, df["f1_好评"], width, label="Positive F1")

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Comparison (Binary)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
