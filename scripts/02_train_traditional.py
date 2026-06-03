"""Traditional ML (binary): TF-IDF -> NB / SVM / XGBoost
Usage: python scripts/02_train_traditional.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
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


def build_features(X_train_texts, X_test_texts, max_features=8000):
    """Build word-level + char-level TF-IDF features and concatenate"""
    # Word-level (jieba tokenized)
    print("  Building word-level TF-IDF...")
    vec_word = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_word = vec_word.fit_transform(X_train_texts)
    X_test_word = vec_word.transform(X_test_texts)
    print(f"  Word TF-IDF dim: {X_train_word.shape[1]}")

    # Char-level (scikit-learn built-in, much faster)
    print("  Building char-level TF-IDF...")
    vec_char = TfidfVectorizer(max_features=max_features, analyzer="char", ngram_range=(1, 3))
    X_train_char = vec_char.fit_transform(X_train_texts)
    X_test_char = vec_char.transform(X_test_texts)
    print(f"  Char TF-IDF dim: {X_train_char.shape[1]}")

    combined = hstack([X_train_word, X_train_char])
    combined_test = hstack([X_test_word, X_test_char])
    print(f"  Combined dim: {combined.shape[1]}")
    return combined, combined_test, vec_word


def tune_xgboost(X_train, y_train):
    """Search on subset, then train final model on full data with best params"""
    from sklearn.model_selection import train_test_split
    scorer = make_scorer(f1_score, average="macro")

    # Sub-sample for fast hyperparameter search
    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train, train_size=15000, stratify=y_train, random_state=42)

    param_dist = {
        "n_estimators": [200, 300, 500],
        "max_depth": [6, 8, 10],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 5.0],
    }

    xgb = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)
    search = RandomizedSearchCV(
        xgb, param_distributions=param_dist, n_iter=20,
        scoring=scorer, cv=3, random_state=42, n_jobs=1, verbose=2
    )
    search.fit(X_sub, y_sub)

    print(f"\n  Best params: {search.best_params_}")
    print(f"  Best CV macro-F1 (subset): {search.best_score_:.4f}")

    # Train final model on full data with best params
    best_params = search.best_params_.copy()
    final_model = XGBClassifier(
        random_state=42, eval_metric="logloss", n_jobs=-1, **best_params)
    final_model.fit(X_train, y_train)
    print(f"  Final model trained on {X_train.shape[0]:,} samples")

    return final_model, search.best_params_


def main():
    parser = argparse.ArgumentParser(description="Traditional ML training")
    parser.add_argument("--balanced", action="store_true",
                        help="Enable class weights for imbalanced data")
    parser.add_argument("--tune-xgboost", action="store_true",
                        help="Use char+word features and RandomizedSearchCV for XGBoost")
    args = parser.parse_args()

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

    # Optional: build char+word combined features for XGBoost tuning
    xgb_train_feat, xgb_test_feat = X_train_tfidf, X_test_tfidf
    if args.tune_xgboost:
        print("\n=== Building char+word features for XGBoost ===")
        combined_train, combined_test, _ = build_features(
            X_train_raw, X_test_raw, max_features=TFIDF_MAX_FEATURES)
        xgb_train_feat, xgb_test_feat = combined_train, combined_test

    xgb_fit_kwargs = {}

    if args.balanced:
        from sklearn.utils.class_weight import compute_sample_weight
        xgb_fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train)
        print("Using class weights for imbalanced training")

    if args.tune_xgboost:
        print("\n=== XGBoost Hyperparameter Search ===")
        xgb_model, _ = tune_xgboost(xgb_train_feat, y_train)
    else:
        xgb_model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=SEED, eval_metric="logloss")

    linear_svc_kwargs = {"C": 1.0, "max_iter": 2000, "random_state": SEED, "dual": False}
    if args.balanced:
        linear_svc_kwargs["class_weight"] = "balanced"

    models = {
        "MultinomialNB": MultinomialNB(alpha=0.5),
        "LinearSVC": CalibratedClassifierCV(LinearSVC(**linear_svc_kwargs)),
        "XGBoost": xgb_model,
    }

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        if name == "XGBoost":
            if xgb_fit_kwargs:
                model.fit(xgb_train_feat, y_train, **xgb_fit_kwargs)
            else:
                model.fit(xgb_train_feat, y_train)
            y_pred = model.predict(xgb_test_feat)
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
