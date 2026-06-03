"""DL from scratch (binary): TextCNN + BiGRU-Attention
Usage: python scripts/03_train_dl.py [--model textcnn|bigru_attn|all]
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

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(csv_path: str, balanced: bool = False) -> torch.Tensor | None:
    if not balanced:
        return None
    from sklearn.utils.class_weight import compute_class_weight
    df = pd.read_csv(csv_path)
    y = df["label"].values
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    print(f"Class weights: neg={weights[0]:.3f}, pos={weights[1]:.3f}")
    return torch.tensor(weights, dtype=torch.float32)


def create_train_loader(csv_path, word2idx, batch_size, max_len, balanced=False):
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
            num_samples=len(dataset), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        from src.dataset import create_data_loader
        return create_data_loader(csv_path, word2idx, batch_size, max_len, shuffle=True)


def train_model(model_type: str, embedding_matrix, word2idx, class_weights, balanced: bool = False):
    print(f"\n{'='*50}")
    print(f"Training {model_type}")

    train_loader = create_train_loader(TRAIN_PATH, word2idx, BATCH_SIZE_DL, MAX_SEQ_LEN, balanced=balanced)
    val_loader = create_data_loader(VAL_PATH, word2idx, BATCH_SIZE_DL, MAX_SEQ_LEN, shuffle=False)
    test_loader = create_data_loader(TEST_PATH, word2idx, BATCH_SIZE_DL, MAX_SEQ_LEN, shuffle=False)

    pretrained = torch.tensor(embedding_matrix, dtype=torch.float32)

    if model_type == "textcnn":
        model = TextCNN(vocab_size=len(word2idx), embed_dim=EMBEDDING_DIM,
                        num_classes=NUM_CLASSES, dropout=DROPOUT,
                        pretrained_embeddings=pretrained, freeze_embeddings=False)
    else:
        model = BiGRUAttention(vocab_size=len(word2idx), embed_dim=EMBEDDING_DIM,
                                num_classes=NUM_CLASSES, dropout=DROPOUT,
                                pretrained_embeddings=pretrained, freeze_embeddings=False)

    save_path = CHECKPOINT_DIR / f"{model_type}_best.pth"
    history, test_metrics = train_loop(
        model, train_loader, val_loader, test_loader,
        epochs=EPOCHS_DL, lr=LR_DL, device=DEVICE,
        save_path=str(save_path), class_weights=class_weights)

    print_metrics(test_metrics)
    append_to_results_csv(RESULTS_PATH, model_type, test_metrics)

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
    parser.add_argument("--model", type=str, default="all", choices=["textcnn", "bigru_attn", "all"])
    parser.add_argument("--wv-path", type=str, default=None, help="Pretrained Word2Vec path")
    parser.add_argument("--balanced", action="store_true",
                        help="Enable WeightedRandomSampler + weighted loss for imbalanced data")
    args = parser.parse_args()

    print("Building vocabulary...")
    word2idx = build_vocab_from_csv(TRAIN_PATH, min_freq=2)
    embedding_matrix = build_embedding_matrix(word2idx, args.wv_path, EMBEDDING_DIM)
    class_weights = compute_class_weights(TRAIN_PATH, balanced=args.balanced)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if args.model in ("textcnn", "all"):
        train_model("textcnn", embedding_matrix, word2idx, class_weights, balanced=args.balanced)
    if args.model in ("bigru_attn", "all"):
        train_model("bigru_attn", embedding_matrix, word2idx, class_weights, balanced=args.balanced)


if __name__ == "__main__":
    main()
