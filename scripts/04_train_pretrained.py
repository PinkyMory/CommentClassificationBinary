"""
预训练模型微调脚本（Step 4）
=============================================================================
对预训练中文 NLP 模型进行情感分类微调。

支持的模型：
  - bert-base-chinese:          Google 发布的中文 BERT（12层，110M 参数）
  - hfl/chinese-roberta-wwm-ext: 哈工大中文 RoBERTa，使用全词遮蔽（WWM），
    在大规模中文语料上预训练，通常效果最佳

微调策略：
  - 在 [CLS] token 输出上加分类头（AutoModelForSequenceClassification）
  - 使用较小的学习率（2e-5）保护预训练权重
  - Warmup + Linear Decay 学习率调度
  - EarlyStopping（patience=2），避免过拟合
  - FP16 混合精度训练（GPU 可用时自动启用，节省显存 + 加速）

模型缓存：
  load_model_and_tokenizer() 优先从本地缓存加载（local_files_only=True），
  失败后自动从 HuggingFace Hub 下载。首次运行需要网络连接和较长的下载时间，
  后续运行直接使用本地缓存，无需网络。

平衡模式（--balanced）：
  通过 WeightedTrainer 使用加权 CrossEntropyLoss，少数类的损失权重更大。

模型保存：
  - roberta_best/ → checkpoints/（含 model + tokenizer + config）
  - bert_best/    → checkpoints/

使用方法：
  python scripts/04_train_pretrained.py                       # 微调两个模型
  python scripts/04_train_pretrained.py --model bert          # 只微调 BERT
  python scripts/04_train_pretrained.py --model roberta       # 只微调 RoBERTa
  python scripts/04_train_pretrained.py --balanced            # 不平衡数据模式
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import warnings
warnings.filterwarnings("ignore")  # 忽略 transformers 的冗余警告信息

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from src.config import (
    SEED, TRAIN_PATH, VAL_PATH, TEST_PATH,
    CHECKPOINT_DIR, RESULTS_PATH, FIGURE_DIR,
    BERT_MODEL_NAME, ROBERTA_MODEL_NAME,
    BATCH_SIZE_PRETRAINED, EPOCHS_PRETRAINED, LR_PRETRAINED,
    MAX_SEQ_LEN_BERT, NUM_CLASSES, LABEL_MAP,
)
from src.preprocess import clean_text
from src.metrics import compute_metrics, print_metrics, append_to_results_csv, save_metrics_to_file
from src.plot import plot_confusion_matrix, plot_training_curves
import matplotlib.pyplot as plt

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WeightedTrainer(Trainer):
    """带类别权重的 Trainer 子类

    覆写 compute_loss 方法，在模型处于训练模式时使用加权 CrossEntropyLoss。
    这样少数类（差评）的分类错误受到更大的惩罚，缓解类别不均衡问题。

    为什么继承 Trainer 而非在 TrainingArguments 中设置？
      transformers 的 Trainer 原生不直接支持 per-class loss weights。
      通过覆写 compute_loss 是官方推荐的自定义损失函数的方式。
    """

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """计算带类别权重的损失

        注意：只在 model.training 时使用加权 loss。
        评估阶段始终使用无加权 loss，保证评估的公平性（与 train_utils.py 一致）。
        """
        labels = inputs.pop("labels")         # 从 inputs 中取出标签
        outputs = model(**inputs)              # 前向传播
        logits = outputs.logits

        if self.class_weights is not None and model.training:
            # 训练模式 + 有类别权重 → 使用加权 CrossEntropyLoss
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            # 评估模式或无权重 → 使用标准 CrossEntropyLoss
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics_fn(pred):
    """HuggingFace Trainer 用的评估函数

    Trainer 每个 epoch 结束后在验证集上调用此函数，
    返回的 "macro_f1" 用作 best_model 的选择标准。

    Args:
        pred: EvalPrediction 对象，包含 predictions（logits）和 label_ids

    Returns:
        {"accuracy": ..., "macro_f1": ...} 字典
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)  # logits → 类别索引
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def load_csv_as_dataset(csv_path: str) -> Dataset:
    """读取 CSV 并转为 HuggingFace Dataset 格式

    HuggingFace Trainer 的 train_dataset/eval_dataset 参数
    接受 datasets.Dataset 对象。此函数完成 pandas → Dataset 的转换。

    注意：在此阶段只做 clean_text（去除 HTML/URL/特殊符），
    不做 jieba 分词。真实的分词/子词切分由预训练 tokenizer 在 tokenize_fn 中完成。
    """
    df = pd.read_csv(csv_path)
    df["text"] = df["text"].astype(str).apply(clean_text)
    return Dataset.from_pandas(df[["text", "label"]])


def train_model(model_name: str, model_path: str, balanced: bool = False):
    """微调单个预训练模型

    完整流程：
      1. 加载并清洗数据
      2. 加载模型和 tokenizer（优先本地缓存）
      3. Tokenize 数据集（截断 + 补齐到 MAX_SEQ_LEN_BERT）
      4. 配置 TrainingArguments + EarlyStoppingCallback
      5. 训练
      6. 在测试集评估
      7. 保存模型 + 图表 + 报告

    Args:
        model_name:  模型标识名（"bert" 或 "roberta"）
        model_path:  HuggingFace 模型路径（如 "bert-base-chinese"）
        balanced:    是否启用加权 loss
    """
    print(f"\n{'='*50}")
    print(f"Fine-tuning {model_name}")

    # ---- 加载数据 ----
    ds_train = load_csv_as_dataset(TRAIN_PATH)
    ds_val = load_csv_as_dataset(VAL_PATH)
    ds_test = load_csv_as_dataset(TEST_PATH)

    # ---- 加载模型和 tokenizer ----
    def load_model_and_tokenizer(model_path):
        """尝试从本地缓存加载，失败则从 HuggingFace Hub 下载

        使用 for local_only in [True, False] 循环：
          第一次：local_only=True → 只从缓存加载，速度快
          第二次：local_only=False → 允许下载（首次运行时）

        这样后续运行无需网络，且首次运行也能正常工作。
        """
        for local_only in [True, False]:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, num_labels=NUM_CLASSES, local_files_only=local_only)
                return tokenizer, model
            except OSError:
                if not local_only:
                    raise  # 两次都失败，抛出异常
                print("  Local cache not found, downloading from HuggingFace...")
                continue

    tokenizer, model = load_model_and_tokenizer(model_path)

    # ---- Tokenize ----
    def tokenize_fn(examples):
        """对 batch 中的文本进行 tokenize

        truncation=True:        超出 max_length 的部分截断
        padding="max_length":   不足的部分补齐到 max_length（保证 batch 内长度一致）
        """
        return tokenizer(examples["text"], truncation=True,
                         padding="max_length", max_length=MAX_SEQ_LEN_BERT)

    # batched=True: 批量处理，利用 tokenizer 的 batch 优化（比逐条快得多）
    ds_train = ds_train.map(tokenize_fn, batched=True)
    ds_val = ds_val.map(tokenize_fn, batched=True)
    ds_test = ds_test.map(tokenize_fn, batched=True)

    # ---- 类别权重（可选） ----
    class_weights = None
    if balanced:
        from sklearn.utils.class_weight import compute_class_weight
        train_labels = pd.read_csv(TRAIN_PATH)["label"].values
        weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_labels)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"Class weights: neg={weights[0]:.3f}, pos={weights[1]:.3f}")

    # ---- 训练参数 ----
    output_dir = CHECKPOINT_DIR / f"{model_name}_intermediate"
    output_dir.mkdir(parents=True, exist_ok=True)

    # warmup_steps: 训练初期学习率从 0 线性增加到 LR_PRETRAINED
    # 取总步数的 10% 作为 warmup，避免训练初期的梯度震荡
    total_steps = (len(ds_train) // BATCH_SIZE_PRETRAINED) * EPOCHS_PRETRAINED
    warmup_steps = int(total_steps * 0.1)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=EPOCHS_PRETRAINED,              # 微调轮数，通常 3-5
        per_device_train_batch_size=BATCH_SIZE_PRETRAINED,  # 训练批大小
        per_device_eval_batch_size=BATCH_SIZE_PRETRAINED * 2,  # 评估批大小可加倍（无梯度，占用显存少）
        learning_rate=LR_PRETRAINED,                     # 2e-5，保护预训练权重
        warmup_steps=warmup_steps,                       # warmup 步数
        weight_decay=0.01,                               # AdamW 权重衰减（L2 正则）
        eval_strategy="epoch",                           # 每个 epoch 结束后评估
        save_strategy="epoch",                           # 每个 epoch 结束后保存
        save_total_limit=1,                              # 只保留最新的 1 个检查点，节省磁盘
        load_best_model_at_end=True,                     # 训练结束后加载最佳模型
        metric_for_best_model="macro_f1",               # 以验证集 macro_f1 为最佳模型标准
        fp16=torch.cuda.is_available(),                 # GPU 可用时启用 FP16 混合精度
        report_to="none",                               # 不上报 wandb/mlflow，避免外部依赖
        seed=SEED,
    )

    # ---- 训练器 ----
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=tokenizer,         # transformers 5.x: 使用 processing_class
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # ---- 测试集评估 ----
    test_preds = trainer.predict(ds_test)
    y_true = test_preds.label_ids
    y_pred = test_preds.predictions.argmax(-1)

    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics)

    # ---- 保存模型 ----
    save_path = CHECKPOINT_DIR / f"{model_name}_best"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"Model saved to {save_path}")

    # ---- 记录结果 ----
    append_to_results_csv(RESULTS_PATH, model_name, metrics)

    # ---- 提取训练历史（将 step 级 loss 聚合为 epoch 级） ----
    # Trainer 的 log_history 按 step 记录，需要手动聚合为 epoch 级
    log_history = trainer.state.log_history

    # 分离训练 loss（仅含 "loss" 不含 "eval_loss" 的条目）
    train_steps = [e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e]
    # 分离验证 log（含 "eval_loss" 的条目）
    eval_logs = [e for e in log_history if "eval_loss" in e]
    val_loss = [e["eval_loss"] for e in eval_logs]
    val_f1 = [e.get("eval_macro_f1", 0) for e in eval_logs]

    # 将训练 step 均分为 epoch 份，取每份的平均值
    n_epochs = len(eval_logs)
    steps_per_epoch = len(train_steps) // n_epochs if n_epochs > 0 else len(train_steps)
    train_epoch_losses = []
    for i in range(n_epochs):
        chunk = train_steps[i*steps_per_epoch:(i+1)*steps_per_epoch]
        train_epoch_losses.append(sum(chunk)/len(chunk) if chunk else 0.0)

    history = {"train_loss": train_epoch_losses, "val_loss": val_loss, "val_f1": val_f1}

    # ---- 生成图表和报告 ----
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if len(train_epoch_losses) > 0 and len(val_loss) > 0:
        plot_training_curves(history, str(FIGURE_DIR / f"{model_name}_training_curves.png"))
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]),
                          ["差评", "好评"],
                          save_path=str(FIGURE_DIR / f"{model_name}_confusion_matrix.png"),
                          title=f"{model_name} CM")
    save_metrics_to_file(metrics, str(FIGURE_DIR / f"{model_name}_report.txt"), model_name)
    plt.close("all")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                        choices=["bert", "roberta", "all"],
                        help="Which model to fine-tune（要微调的模型）")
    parser.add_argument("--balanced", action="store_true",
                        help="Enable weighted CrossEntropyLoss（启用加权损失函数）")
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if args.model in ("bert", "all"):
        train_model("bert", BERT_MODEL_NAME, balanced=args.balanced)
    if args.model in ("roberta", "all"):
        train_model("roberta", ROBERTA_MODEL_NAME, balanced=args.balanced)


if __name__ == "__main__":
    main()
