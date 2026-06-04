"""
传统机器学习训练脚本（Step 2）
=============================================================================
使用 TF-IDF 特征 + 三种传统机器学习模型进行二分类。

模型列表：
  - MultinomialNB（多项式朴素贝叶斯）: 简单高效，适合高维稀疏特征
  - LinearSVC（线性支持向量机）: 最大间隔分类，配合 CalibratedClassifierCV
    获得概率输出
  - XGBoost（梯度提升树）: 强大的集成学习方法

特征提取：
  默认模式：jieba 分词 + TF-IDF (unigram+bigram, 5000维)
  调优模式（--tune-xgboost）：word级TF-IDF(5000维) + char级TF-IDF(5000维)
                              = 10000维组合特征 + RandomizedSearchCV

特殊参数：
  --balanced:   启用类别权重（SVM class_weight + XGBoost sample_weight）
  --tune-xgboost: 对 XGBoost 启用组合特征 + 超参数搜索

模型保存：
  - MultinomialNB → checkpoints/multinomialnb.pkl
  - LinearSVC     → checkpoints/linearsvc.pkl
  - XGBoost       → checkpoints/xgboost.json
  - TF-IDF 向量器 → checkpoints/tfidf_vectorizer.pkl（供 demo 推理使用）

使用方法：
  python scripts/02_train_traditional.py                         # 默认模式
  python scripts/02_train_traditional.py --tune-xgboost          # XGBoost 调优
  python scripts/02_train_traditional.py --balanced              # 类别权重（不均衡数据）
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
    """对文本列表进行 jieba 分词并用空格连接

    为什么要用空格连接？
      sklearn 的 TfidfVectorizer 默认以空格分隔的词元为单位。
      用空格连接后，TfidfVectorizer 会自动按空格切分得到 jieba 分词结果。

    Args:
        texts: 原始评论文本列表

    Returns:
        ["词1 词2 词3", ...] 格式的字符串列表
    """
    return [" ".join(jieba.cut(clean_text(t))) for t in texts]


def build_features(X_train_texts, X_test_texts, max_features=8000):
    """构建 word-level + char-level 组合 TF-IDF 特征

    组合策略：
      - 词级 TF-IDF: jieba 分词后提取 unigram+bigram（5000维）
      - 字级 TF-IDF: 直接在原始文本上提取 1/2/3-gram（5000维）
      - 用 scipy.sparse.hstack 水平拼接 → 10000维

    为什么加字级特征？
      中文的字级 n-gram 能捕捉字形级别的模式，对未登录词（OOV）具有鲁棒性。
      例如"差评"这个词即使 jieba 没切出来，字级 3-gram 可以捕获到。

    Args:
        X_train_texts: 训练集原始文本列表
        X_test_texts:  测试集原始文本列表
        max_features:  每种特征的最大维度

    Returns:
        (X_train_combined, X_test_combined, vec_word):
          - 训练和测试的稀疏特征矩阵
          - 词级向量器（供 demo 推理时重建特征）
    """
    # ---- 词级 TF-IDF（jieba 分词后） ----
    print("  Building word-level TF-IDF...")
    # 先 jieba 分词再用空格连接
    X_train_tokenized = tokenize_and_join(X_train_texts)
    X_test_tokenized = tokenize_and_join(X_test_texts)
    vec_word = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_word = vec_word.fit_transform(X_train_tokenized)
    X_test_word = vec_word.transform(X_test_tokenized)
    print(f"  Word TF-IDF dim: {X_train_word.shape[1]}")

    # ---- 字级 TF-IDF（直接用原始文本，scikit-learn 内置 analyzer="char"） ----
    print("  Building char-level TF-IDF...")
    # analyzer="char" 表示以字符为单元（而非词语），ngram_range=(1,3) 为 1/2/3-gram
    # 这比手动构建字级 tokenizer 快得多
    vec_char = TfidfVectorizer(max_features=max_features, analyzer="char", ngram_range=(1, 3))
    X_train_char = vec_char.fit_transform(X_train_texts)
    X_test_char = vec_char.transform(X_test_texts)
    print(f"  Char TF-IDF dim: {X_train_char.shape[1]}")

    # ---- 水平拼接 ----
    combined = hstack([X_train_word, X_train_char])
    combined_test = hstack([X_test_word, X_test_char])
    print(f"  Combined dim: {combined.shape[1]}")
    return combined, combined_test, vec_word


def tune_xgboost(X_train, y_train):
    """XGBoost 超参数随机搜索

    搜索策略：
      1. 从训练集中抽样 15000 条用于快速搜索（避免在全量数据上搜索太慢）
      2. 在抽样数据上做 RandomizedSearchCV（20 组参数组合，3 折交叉验证）
      3. 找到最佳参数后，在全量训练数据上重新训练最终模型

    为什么用 RandomizedSearch 而非 GridSearch？
      GridSearch 在 7 个参数上会组合出上千种配置，对 XGBoost 完全不可行。
      RandomizedSearch 随机采样 20 组，通常能找到接近最优的参数。

    搜索参数空间说明：
      - n_estimators:  树的数量，越多越强但越慢
      - max_depth:     树的最大深度，越大越容易过拟合
      - learning_rate: 学习率，越小需要越多的 n_estimators
      - subsample:     每棵树使用的样本比例（行采样）
      - colsample_bytree: 每棵树使用的特征比例（列采样）
      - reg_alpha:     L1 正则化（稀疏性正则）
      - reg_lambda:    L2 正则化（权重衰减正则）
    """
    from sklearn.model_selection import train_test_split
    # 用 macro_f1 作为搜索评分指标
    scorer = make_scorer(f1_score, average="macro")

    # 从训练集中抽样以加速搜索
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

    # 用最佳参数在全量数据上训练最终模型
    best_params = search.best_params_.copy()
    final_model = XGBClassifier(
        random_state=42, eval_metric="logloss", n_jobs=-1, **best_params)
    final_model.fit(X_train, y_train)
    print(f"  Final model trained on {X_train.shape[0]:,} samples")

    return final_model, search.best_params_


def main():
    parser = argparse.ArgumentParser(description="Traditional ML training")
    parser.add_argument("--balanced", action="store_true",
                        help="Enable class weights for imbalanced data（为不均衡数据启用类别权重）")
    parser.add_argument("--tune-xgboost", action="store_true",
                        help="Use char+word features and RandomizedSearchCV for XGBoost（对 XGBoost 使用组合特征+超参搜索）")
    args = parser.parse_args()

    # ---- 加载数据 ----
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # 合并 train+val 用于传统 ML 训练（传统 ML 不需要验证集，数据越多越好）
    train_all = pd.concat([train_df, val_df], ignore_index=True)
    X_train_raw = train_all["text"].tolist()
    y_train = train_all["label"].tolist()
    X_test_raw = test_df["text"].tolist()
    y_test = test_df["label"].tolist()

    print(f"Train: {len(X_train_raw):,} rows, Test: {len(X_test_raw):,} rows")

    # ---- TF-IDF 特征提取（默认模式：词级 TF-IDF） ----
    print("\n=== TF-IDF Feature Extraction ===")
    print("Tokenizing...")
    X_train_tokens = tokenize_and_join(X_train_raw)
    X_test_tokens = tokenize_and_join(X_test_raw)

    # TfidfVectorizer: 将文本转为 TF-IDF 稀疏矩阵
    # max_features=5000 限制特征维度，防止维度过大导致过拟合和计算开销
    # ngram_range=(1,2) 同时提取 unigram 和 bigram
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE)
    X_train_tfidf = vectorizer.fit_transform(X_train_tokens)
    X_test_tfidf = vectorizer.transform(X_test_tokens)
    print(f"Feature dim: {X_train_tfidf.shape[1]}")

    # ---- XGBoost 调优模式：构建组合特征 ----
    xgb_train_feat, xgb_test_feat = X_train_tfidf, X_test_tfidf
    if args.tune_xgboost:
        print("\n=== Building char+word features for XGBoost ===")
        combined_train, combined_test, _ = build_features(
            X_train_raw, X_test_raw, max_features=TFIDF_MAX_FEATURES)
        xgb_train_feat, xgb_test_feat = combined_train, combined_test

    # ---- 类别权重（可选） ----
    xgb_fit_kwargs = {}
    if args.balanced:
        from sklearn.utils.class_weight import compute_sample_weight
        # sample_weight: 每条样本的损失权重，少数类样本的权重更大
        xgb_fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train)
        print("Using class weights for imbalanced training")

    # ---- 初始化模型 ----
    # LinearSVC 包装 CalibratedClassifierCV 以获得概率输出
    # 原因：SVM 本身只输出决策值，不输出概率。CalibratedClassifierCV
    # 通过 Platt Scaling（S 型函数拟合）将决策值校准为概率。
    # dual=False 是因为特征维度通常大于样本数（5k 特征 vs 几万样本用 primal 更高效）
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

    # 模型字典：key=模型名, value=模型实例
    models = {
        "MultinomialNB": MultinomialNB(alpha=0.5),  # alpha=0.5: 拉普拉斯平滑参数
        "LinearSVC": CalibratedClassifierCV(LinearSVC(**linear_svc_kwargs)),
        "XGBoost": xgb_model,
    }

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 逐个训练和评估 ----
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")

        # XGBoost 使用可能不同的特征矩阵（调优模式下的组合特征）
        if name == "XGBoost":
            if xgb_fit_kwargs:
                model.fit(xgb_train_feat, y_train, **xgb_fit_kwargs)
            else:
                model.fit(xgb_train_feat, y_train)
            y_pred = model.predict(xgb_test_feat)
        else:
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)

        # 评估
        metrics = compute_metrics(y_test, y_pred)
        print_metrics(metrics)

        # 保存模型（NB/SVM → pickle, XGBoost → JSON）
        ext = "pkl" if name != "XGBoost" else "json"
        save_path = CHECKPOINT_DIR / f"{name.lower()}.{ext}"
        if ext == "pkl":
            with open(save_path, "wb") as f:
                pickle.dump(model, f)
        else:
            # XGBoost 原生支持 save_model/load_model，比 pickle 更稳定
            model.save_model(str(save_path))
        print(f"Model saved to {save_path}")

        # 追加结果到 results.csv
        append_to_results_csv(RESULTS_PATH, name, metrics)

        # 生成混淆矩阵图和评估报告
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        model_key = name.lower()
        cm_path = FIGURE_DIR / f"{model_key}_confusion_matrix.png"
        plot_confusion_matrix(np.array(metrics["confusion_matrix"]),
                              ["差评", "好评"], save_path=str(cm_path), title=f"{name} CM")
        plt.close("all")  # 关闭图形，释放内存

        report_path = FIGURE_DIR / f"{model_key}_report.txt"
        save_metrics_to_file(metrics, str(report_path), name)

    # 保存 TF-IDF 向量器，供 app/demo.py 推理时使用
    vectorizer_path = CHECKPOINT_DIR / "tfidf_vectorizer.pkl"
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"\nTF-IDF Vectorizer saved to {vectorizer_path}")


if __name__ == "__main__":
    main()
