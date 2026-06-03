"""Traditional ML (binary): TF-IDF -> NB / SVM / XGBoost
Usage: python scripts/02_train_traditional.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from src.config import (
    SEED, TRAIN_PATH, VAL_PATH, TEST_PATH,
    CHECKPOINT_DIR, RESULTS_PATH, FIGURE_DIR,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
)
from src.preprocess import clean_text
from src.metrics import compute_metrics, print_metrics, append_to_results_csv, save_metrics_to_file
from src.plot import plot_confusion_matrix
import jieba

np.random.seed(SEED)


def tokenize_and_join(texts: list[str]) -> list[str]:
    """jieba tokenize then join with spaces"""
    return [" ".join(jieba.cut(clean_text(t))) for t in texts]


def main():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Merge train+val for traditional ML training
    train_all = pd.concat([train_df, val_df], ignore_index=True)
    X_train_raw = train_all["text"].tolist()
    y_train = train_all["label"].tolist()
    X_test_raw = test_df["text"].tolist()
    y_test = test_df["label"].tolist()

    print(f"Train: {len(X_train_raw):,} rows, Test: {len(X_test_raw):,} rows")

    # TF-IDF
    print("\n=== TF-IDF Feature Extraction ===")
    print("Tokenizing...")
    X_train_tokens = tokenize_and_join(X_train_raw)
    X_test_tokens = tokenize_and_join(X_test_raw)

    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE)
    X_train_tfidf = vectorizer.fit_transform(X_train_tokens)
    X_test_tfidf = vectorizer.transform(X_test_tokens)
    print(f"Feature dim: {X_train_tfidf.shape[1]}")

    # Models (all with class_weight for imbalance)
    models = {
        "MultinomialNB": MultinomialNB(alpha=0.5),
        "LinearSVC": CalibratedClassifierCV(LinearSVC(
            C=1.0, max_iter=2000, random_state=SEED, dual=False, class_weight="balanced")),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=SEED, eval_metric="logloss"),
    }
    from sklearn.utils.class_weight import compute_sample_weight
    xgb_sample_weight = compute_sample_weight("balanced", y_train)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        if name == "XGBoost":
            model.fit(X_train_tfidf, y_train, sample_weight=xgb_sample_weight)
        else:
            model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        metrics = compute_metrics(y_test, y_pred)
        print_metrics(metrics)

        ext = "pkl" if name != "XGBoost" else "json"
        save_path = CHECKPOINT_DIR / f"{name.lower()}.{ext}"
        if ext == "pkl":
            with open(save_path, "wb") as f:
                pickle.dump(model, f)
        else:
            model.save_model(str(save_path))
        print(f"Model saved to {save_path}")

        append_to_results_csv(RESULTS_PATH, name, metrics)

        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        model_key = name.lower()
        cm_path = FIGURE_DIR / f"{model_key}_confusion_matrix.png"
        plot_confusion_matrix(np.array(metrics["confusion_matrix"]),
                              ["差评", "好评"], save_path=str(cm_path), title=f"{name} CM")
        plt.close("all")

        report_path = FIGURE_DIR / f"{model_key}_report.txt"
        save_metrics_to_file(metrics, str(report_path), name)

    # Save vectorizer for demo
    vectorizer_path = CHECKPOINT_DIR / "tfidf_vectorizer.pkl"
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"\nTF-IDF Vectorizer saved to {vectorizer_path}")


if __name__ == "__main__":
    main()
