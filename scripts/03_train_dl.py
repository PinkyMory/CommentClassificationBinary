"""
深度学习从零训练脚本（Step 3）
=============================================================================
不依赖预训练模型，从头训练 TextCNN 和 BiGRU-Attention 两个模型。

模型架构：
  - TextCNN:          多尺寸卷积核(3/4/5-gram) → MaxPool → Concat → FC
  - BiGRU-Attention:  双向 GRU → Bahdanau 注意力 → FC

训练配置：
  - Embedding: 300维（随机初始化 或 加载预训练Word2Vec）
  - Optimizer: Adam (lr=1e-3) + ReduceLROnPlateau
  - EarlyStopping: patience=5
  - Epochs: 最大 30
  - Batch: 64

词向量初始化：
  1. 默认：正态分布随机初始化（无需外部文件即可运行）
  2. --wv-path: 加载预训练 Word2Vec 词向量，提升效果

平衡模式（--balanced）：
  - WeightedRandomSampler: 少数类样本被更频繁地采样
  - Weighted CrossEntropyLoss: 少数类的分类错误惩罚更大

模型保存：
  - textcnn_best.pth     → checkpoints/
  - bigru_attn_best.pth  → checkpoints/

使用方法：
  python scripts/03_train_dl.py                               # 训练两个模型
  python scripts/03_train_dl.py --model textcnn               # 只训练 TextCNN
  python scripts/03_train_dl.py --model bigru_attn            # 只训练 BiGRU
  python scripts/03_train_dl.py --wv-path data/embeddings/sgns.weibo.word  # 预训练词向量
  python scripts/03_train_dl.py --balanced                    # 不平衡数据模式
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import torch
import numpy as np
import pandas as pd
from src.config import (
    SEED, TRAIN_PATH, VAL_PATH, TEST_PATH,
    CHECKPOINT_DIR, RESULTS_PATH, FIGURE_DIR,
    EMBEDDING_DIM, MAX_SEQ_LEN, NUM_CLASSES,
    BATCH_SIZE_DL, EPOCHS_DL, LR_DL, DROPOUT,
)
from src.dataset import create_data_loader, build_vocab_from_csv, build_embedding_matrix
from src.models.textcnn import TextCNN
from src.models.bigru_attn import BiGRUAttention
from src.train_utils import train_loop
from src.metrics import print_metrics, append_to_results_csv, save_metrics_to_file
from src.plot import plot_confusion_matrix, plot_training_curves
import matplotlib.pyplot as plt

# 固定所有随机源，保证实验可复现
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 自动检测并使用 GPU，不可用时回退到 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(csv_path: str, balanced: bool = False) -> torch.Tensor | None:
    """计算类别权重（用于加权 CrossEntropyLoss）

    只在 --balanced 模式下启用。
    权重计算公式：n_samples / (n_classes * n_samples_per_class)
    少数类的权重 > 1，多数类的权重 < 1。

    Args:
        csv_path:  训练集 CSV 路径
        balanced:  是否启用类别权重

    Returns:
        类别权重张量 [neg_weight, pos_weight]，未启用时返回 None
    """
    if not balanced:
        return None
    from sklearn.utils.class_weight import compute_class_weight
    df = pd.read_csv(csv_path)
    y = df["label"].values
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    print(f"Class weights: neg={weights[0]:.3f}, pos={weights[1]:.3f}")
    return torch.tensor(weights, dtype=torch.float32)


def create_train_loader(csv_path, word2idx, batch_size, max_len, balanced=False):
    """创建训练数据加载器，支持 WeightedRandomSampler

    WeightedRandomSampler 的工作原理：
      每个样本的采样概率与其类别权重成正比。
      少数类样本权重更大 → 被采样更频繁 → 每个 batch 中两类样本数趋于均衡。

      这比直接修改 loss 更"物理"地均衡了训练过程，但可能导致某些样本
      在单个 epoch 中被多次采样，增加潜在的过拟合风险。

    Args:
        csv_path:  训练集 CSV 路径
        word2idx:  词表
        batch_size: 批大小
        max_len:   最大序列长度
        balanced:  是否启用加权采样

    Returns:
        DataLoader 实例
    """
    from src.dataset import TokenizedDataset
    dataset = TokenizedDataset(csv_path, word2idx, max_len)
    if balanced:
        from torch.utils.data import WeightedRandomSampler, DataLoader
        df = pd.read_csv(csv_path)
        labels = df["label"].values
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight("balanced", labels)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float64),
            num_samples=len(dataset),
            replacement=True  # 有放回采样，少数类样本会被重复采样
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        from src.dataset import create_data_loader
        return create_data_loader(csv_path, word2idx, batch_size, max_len, shuffle=True)


def train_model(model_type: str, embedding_matrix, word2idx, class_weights, balanced: bool = False):
    """训练单个深度学习模型（TextCNN 或 BiGRU-Attention）

    完整流程：
      1. 创建数据加载器（训练/验证/测试）
      2. 初始化模型（加载预训练词向量）
      3. 调用 train_loop 训练（含 EarlyStopping 和最佳模型保存）
      4. 评估并保存结果

    Args:
        model_type:       "textcnn" 或 "bigru_attn"
        embedding_matrix: 词向量矩阵 [vocab_size, embed_dim]
        word2idx:         词表
        class_weights:    类别权重（用于加权 loss）
        balanced:         是否启用加权采样
    """
    print(f"\n{'='*50}")
    print(f"Training {model_type}")

    # 创建数据加载器
    # 训练集：shuffle 设为 True（默认），确保每个 epoch 数据顺序随机
    # 验证/测试集：shuffle=False，保证评估结果一致
    train_loader = create_train_loader(TRAIN_PATH, word2idx, BATCH_SIZE_DL, MAX_SEQ_LEN, balanced=balanced)
    val_loader = create_data_loader(VAL_PATH, word2idx, BATCH_SIZE_DL, MAX_SEQ_LEN, shuffle=False)
    test_loader = create_data_loader(TEST_PATH, word2idx, BATCH_SIZE_DL, MAX_SEQ_LEN, shuffle=False)

    # 将 numpy 矩阵转为 torch 张量供 Embedding 层使用
    pretrained = torch.tensor(embedding_matrix, dtype=torch.float32)

    # 根据 model_type 初始化对应模型
    if model_type == "textcnn":
        model = TextCNN(
            vocab_size=len(word2idx), embed_dim=EMBEDDING_DIM,
            num_classes=NUM_CLASSES, dropout=DROPOUT,
            pretrained_embeddings=pretrained, freeze_embeddings=False)
    else:
        model = BiGRUAttention(
            vocab_size=len(word2idx), embed_dim=EMBEDDING_DIM,
            num_classes=NUM_CLASSES, dropout=DROPOUT,
            pretrained_embeddings=pretrained, freeze_embeddings=False)

    # 训练
    save_path = CHECKPOINT_DIR / f"{model_type}_best.pth"
    history, test_metrics = train_loop(
        model, train_loader, val_loader, test_loader,
        epochs=EPOCHS_DL, lr=LR_DL, device=DEVICE,
        save_path=str(save_path), class_weights=class_weights)

    # 评估和保存
    print_metrics(test_metrics)
    append_to_results_csv(RESULTS_PATH, model_type, test_metrics)

    # 生成图表和报告
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history, str(FIGURE_DIR / f"{model_type}_training_curves.png"))
    plot_confusion_matrix(np.array(test_metrics["confusion_matrix"]),
                          ["差评", "好评"],
                          save_path=str(FIGURE_DIR / f"{model_type}_confusion_matrix.png"),
                          title=f"{model_type} CM")
    save_metrics_to_file(test_metrics, str(FIGURE_DIR / f"{model_type}_report.txt"), model_type)
    plt.close("all")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                        choices=["textcnn", "bigru_attn", "all"],
                        help="Which model to train（要训练的模型）")
    parser.add_argument("--wv-path", type=str, default=None,
                        help="Pretrained Word2Vec path（预训练词向量路径，可选）")
    parser.add_argument("--balanced", action="store_true",
                        help="Enable WeightedRandomSampler + weighted loss（启用不平衡数据处理）")
    args = parser.parse_args()

    # ---- 构建词表和词向量矩阵 ----
    print("Building vocabulary...")
    # min_freq=2: 过滤只出现1次的低频词，减少词表噪音
    word2idx = build_vocab_from_csv(TRAIN_PATH, min_freq=2)
    # 构建词向量矩阵：有预训练向量则加载，否则随机初始化
    embedding_matrix = build_embedding_matrix(word2idx, args.wv_path, EMBEDDING_DIM)
    # 可选的类别权重
    class_weights = compute_class_weights(TRAIN_PATH, balanced=args.balanced)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 训练模型 ----
    if args.model in ("textcnn", "all"):
        train_model("textcnn", embedding_matrix, word2idx, class_weights, balanced=args.balanced)
    if args.model in ("bigru_attn", "all"):
        train_model("bigru_attn", embedding_matrix, word2idx, class_weights, balanced=args.balanced)


if __name__ == "__main__":
    main()
