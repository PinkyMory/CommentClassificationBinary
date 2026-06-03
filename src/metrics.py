from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
import pandas as pd
import numpy as np


def compute_metrics(y_true, y_pred, label_names=None) -> dict:
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
    """Formatted print of evaluation results"""
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
    """Append (or replace if exists) model results to results.csv"""
    import os
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
        df_existing = pd.read_csv(results_csv_path)
        df_existing = df_existing[df_existing["model"] != model_name]
        df_out = pd.concat([df_existing, df_row], ignore_index=True)
    else:
        df_out = df_row
    df_out.to_csv(results_csv_path, index=False)
    print(f"Results written to {results_csv_path}")


def save_metrics_to_file(metrics: dict, save_path: str, model_name: str = ""):
    """Save evaluation metrics as formatted text file"""
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
