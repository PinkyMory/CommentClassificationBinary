"""
深度学习训练工具模块
=============================================================================
为 TextCNN 和 BiGRU-Attention 的训练提供通用组件。

主要类/函数：
  - EarlyStopping: 早停机制，防止过拟合
  - train_epoch:   单 epoch 训练（前向 + 反向传播 + 参数更新）
  - evaluate:      在给定数据集上评估模型（无梯度计算）
  - train_loop:    完整的训练循环（训练 + 验证 + 测试 + 模型保存）

注意事项：
  - evaluate 中始终使用无加权的 CrossEntropyLoss 进行评估，
    即使训练使用了 class_weights。评估阶段的损失仅用于 EarlyStopping 判断，
    与指标的公平性无关。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from src.metrics import compute_metrics


class EarlyStopping:
    """早停（Early Stopping）机制

    监控验证集损失，当连续 patience 个 epoch 没有显著改善时停止训练。
    显著改善的定义：当前 val_loss 比历史最佳降低了至少 min_delta。

    使用方法：
      stopper = EarlyStopping(patience=5)
      for epoch in range(epochs):
          val_loss = validate(...)
          if stopper(val_loss):
              break  # 触发早停

    为什么用 val_loss 而非 val_f1 来判断？
      - loss 是连续且平滑的信号，比 F1 更适合作为早停判据
      - F1 是离散指标的跳跃变化，小 batch 下的波动可能触发误判
      - 但最终保存的是 val_f1 最高的模型，而非 val_loss 最低的模型
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        """
        Args:
            patience:  容忍的 epoch 数，超过此数未改善则停止
            min_delta: 最小改善阈值，改善小于此值视为无改善（防止微小波动重置计数器）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0               # 当前连续未改善的 epoch 数
        self.best_loss = float("inf")  # 历史最佳验证损失
        self.early_stop = False        # 是否触发早停

    def __call__(self, val_loss: float) -> bool:
        """传入当前 epoch 的验证损失，返回是否应停止训练"""
        if val_loss < self.best_loss - self.min_delta:
            # 有足够改善：重置计数器，更新最佳损失
            self.best_loss = val_loss
            self.counter = 0
        else:
            # 未改善：计数器 +1
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_epoch(model, dataloader, optimizer, criterion, device) -> float:
    """执行一个 epoch 的训练（包含前向传播、反向传播和参数更新）

    流程：
      1. 将模型设为 train 模式（启用 Dropout 等）
      2. 逐 batch 进行前向传播 → 计算损失 → 反向传播 → 优化器更新
      3. 返回该 epoch 的平均损失（按样本加权平均）

    Args:
        model:      PyTorch 模型（需接收 (input_ids, attention_mask) 并返回 logits）
        dataloader: 训练数据加载器
        optimizer:  优化器（Adam）
        criterion:  损失函数（CrossEntropyLoss，可能带 class_weights）
        device:     计算设备（"cuda" 或 "cpu"）

    Returns:
        该 epoch 的平均训练损失
    """
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        # 将数据移动到指定设备
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()               # 清零梯度（防止累积）
        logits = model(input_ids, attention_mask)  # 前向传播
        loss = criterion(logits, labels)     # 计算损失
        loss.backward()                      # 反向传播
        optimizer.step()                     # 更新参数

        # 按样本数加权累加损失，保证不同 batch_size 下的可比性
        total_loss += loss.item() * input_ids.size(0)

    # 返回整个 epoch 的平均损失
    return total_loss / len(dataloader.dataset)


@torch.no_grad()  # 禁用梯度计算，减少显存占用并加速
def evaluate(model, dataloader, criterion, device) -> tuple[float, dict]:
    """在数据集上评估模型

    与训练不同：
      - 使用 model.eval() 模式（Dropout 失效，BatchNorm 使用全局统计量）
      - 使用 @torch.no_grad() 上下文（不构建计算图）
      - 评估用的 loss 始终是无加权的 CrossEntropyLoss（保证评估公平性）

    Args:
        model:      PyTorch 模型
        dataloader: 验证/测试数据加载器
        criterion:  训练用的损失函数（此处不使用，内部重建无加权版本）
        device:     计算设备

    Returns:
        (avg_loss, metrics): 平均损失 和 compute_metrics() 返回的指标字典
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    # 始终使用无加权的 CrossEntropyLoss 进行评估
    # 这样 val_loss 反映模型在均衡分布下的真实损失，不受 class_weights 影响
    eval_criterion = nn.CrossEntropyLoss()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = eval_criterion(logits, labels)
        total_loss += loss.item() * input_ids.size(0)

        # 获取预测类别：logits 中概率最大的那个
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics


def train_loop(model, train_loader, val_loader, test_loader=None,
               epochs=30, lr=1e-3, device=None, save_path=None,
               patience=5, class_weights=None):
    """完整的训练、验证、测试循环

    包含：
      - 优化器（Adam + ReduceLROnPlateau 学习率调度）
      - EarlyStopping（patience=5 个 epoch）
      - 最佳模型保存（按 val_macro_f1 最高）
      - 训练历史记录（loss 和 F1 曲线）
      - 可选：训练结束后在测试集上评估最佳模型

    学习率调度策略：
      使用 ReduceLROnPlateau，当 val_loss 连续 2 个 epoch 不下降时，
      学习率减半。这比固定学习率能收敛到更好的局部最优。

    Args:
        model:         PyTorch 模型
        train_loader:  训练数据加载器
        val_loader:    验证数据加载器
        test_loader:   测试数据加载器（可选）
        epochs:        最大训练轮数
        lr:            初始学习率
        device:        计算设备
        save_path:     最佳模型保存路径
        patience:      EarlyStopping 的耐心值
        class_weights: 类别权重张量（用于不平衡训练）

    Returns:
        (history, test_metrics):
          - history: 包含 train_loss, val_loss, val_f1 列表的字典
          - test_metrics: 测试集评估指标（无 test_loader 时为 None）
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 如果指定了 class_weights，将其移到目标设备
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ReduceLROnPlateau：当 val_loss 不再下降时自动降低学习率
    # factor=0.5 表示减半，patience=2 表示连续 2 个 epoch 无改善则触发
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)

    early_stop = EarlyStopping(patience=patience)
    best_val_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        # 训练一个 epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        # 在验证集上评估
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        # 更新学习率
        scheduler.step(val_loss)

        # 记录历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_metrics["macro_f1"])

        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f}")

        # 保存验证集 macro_f1 最高的模型
        if val_metrics["macro_f1"] > best_val_f1 and save_path:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), save_path)
            print(f"  => Saved best model to {save_path}")

        # 检查是否早停
        if early_stop(val_loss):
            print(f"  Early stopping at epoch {epoch}")
            break

    # 训练结束后，加载最佳模型并在测试集上评估
    if test_loader and save_path:
        model.load_state_dict(torch.load(save_path, map_location=device))
        test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
        print(f"\nTest Loss: {test_loss:.4f} | Test Macro-F1: {test_metrics['macro_f1']:.4f}")
        return history, test_metrics

    return history, None
