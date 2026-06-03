import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from src.metrics import compute_metrics


class EarlyStopping:
    """Stops when val_loss doesn't improve for 'patience' consecutive epochs"""
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_epoch(model, dataloader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    eval_criterion = nn.CrossEntropyLoss()  # unweighted for validation
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        logits = model(input_ids, attention_mask)
        loss = eval_criterion(logits, labels)
        total_loss += loss.item() * input_ids.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics


def train_loop(model, train_loader, val_loader, test_loader=None,
               epochs=30, lr=1e-3, device=None, save_path=None,
               patience=5, class_weights=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)
    early_stop = EarlyStopping(patience=patience)
    best_val_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_metrics["macro_f1"])
        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_val_f1 and save_path:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), save_path)
            print(f"  => Saved best model to {save_path}")
        if early_stop(val_loss):
            print(f"  Early stopping at epoch {epoch}")
            break

    if test_loader and save_path:
        model.load_state_dict(torch.load(save_path, map_location=device))
        test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
        print(f"\nTest Loss: {test_loss:.4f} | Test Macro-F1: {test_metrics['macro_f1']:.4f}")
        return history, test_metrics
    return history, None
