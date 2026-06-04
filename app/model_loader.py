"""
统一模型加载与推理接口
=============================================================================
提供 SentimentPredictor 类，封装所有模型类型的加载和推理逻辑。

支持的模型类型：
  - 传统 ML:  naive_bayes (MultinomialNB), svm (LinearSVC), xgboost
  - DL 从零:  textcnn, bigru_attn
  - 预训练:    bert, roberta

统一的推理接口：
  >>> predictor = SentimentPredictor("roberta", "checkpoints/roberta_best")
  >>> label, probs = predictor.predict("产品质量很好")
  >>> # label: "好评", probs: [0.12, 0.88]

设计要点：
  1. 所有模型类型对外暴露同一个 predict() 方法
  2. 模型加载在 __init__ 中完成（一次性）
  3. 不同模型类型使用不同的加载策略：
     - 传统 ML: pickle 反序列化 (+ TfidfVectorizer)
     - XGBoost: 原生 load_model + 自动检测是否需要重建组合特征向量器
     - DL: torch.load_state_dict + 重建模型架构
     - 预训练: AutoModel.from_pretrained + AutoTokenizer

XGBoost 调优模式检测：
  如果 XGBoost 模型在加载时未提供 vectorizer，说明是调优模式（10000维组合特征）。
  此时需要根据训练阶段的参数重建 word-level 和 char-level 的 TfidfVectorizer，
  并在 predict 时水平拼接两种特征。
"""

import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentPredictor:
    """统一的模型预测器

    加载任意已训练模型，通过 predict(text) → (label, probs) 接口进行推理。
    """

    def __init__(self, model_type: str, model_path: str, **kwargs):
        """
        Args:
            model_type: 模型类型标识，取值：
                "naive_bayes" | "svm" | "xgboost" |
                "textcnn" | "bigru_attn" |
                "bert" | "roberta"
            model_path: 模型权重/检查点路径
            **kwargs:   额外资源（如 vectorizer, word2idx）
        """
        self.model_type = model_type
        self.model_path = model_path
        self.labels = ["差评", "好评"]  # 二分类标签

        # 根据模型类型分发到不同的加载方法
        if model_type in ("naive_bayes", "svm", "xgboost"):
            self._load_traditional(**kwargs)
        elif model_type in ("textcnn", "bigru_attn"):
            self._load_dl(**kwargs)
        elif model_type in ("bert", "roberta"):
            self._load_pretrained(**kwargs)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # ========================================================================
    # 传统 ML 模型加载
    # ========================================================================

    def _load_traditional(self, **kwargs):
        """加载传统机器学习模型（NB / SVM / XGBoost）

        XGBoost 特殊处理：
          - 默认模式：使用传入的 TfidfVectorizer（5000维词级TF-IDF）
          - 调优模式：未传 vectorizer → 自动重建 word+char 组合特征向量器（10000维）
            （通过 checkpoints 中的 tfidf_vectorizer.pkl + 重建 char 级向量器）
        """
        if self.model_type == "xgboost":
            from xgboost import XGBClassifier
            self.model = XGBClassifier()
            self.model.load_model(self.model_path)  # XGBoost 原生加载，比 pickle 更稳定
            self.vectorizer = kwargs.get("vectorizer")

            if self.vectorizer is None:
                # 调优模式：没有提供标准 vectorizer，需要重建 word+char 组合向量器
                # 这发生在 --tune-xgboost 模式下训练出的模型
                self._build_xgboost_vectorizers()
        else:
            # NB / SVM: 用 pickle 反序列化
            self.model = pickle.load(open(self.model_path, "rb"))
            self.vectorizer = kwargs.get("vectorizer")
            if self.vectorizer is None:
                raise ValueError("Traditional models require vectorizer (TfidfVectorizer)")

    def _build_xgboost_vectorizers(self):
        """为 XGBoost 调优模式重建 word+char 组合特征向量器

        重建逻辑：
          1. 从训练数据重新拟合 word-level TfidfVectorizer（jieba分词 + unigram+bigram, 5000维）
          2. 从训练数据重新拟合 char-level TfidfVectorizer（1/2/3-gram, 5000维）
          3. 存储两个向量器的引用，predict 时水平拼接特征

        注意：重建的向量器需要与训练时的参数完全一致（max_features, ngram_range），
        否则特征维度不匹配会导致预测失败。
        """
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse import hstack
        from src.config import TRAIN_PATH, VAL_PATH, TFIDF_MAX_FEATURES

        # 加载训练数据（train + val，与训练脚本一致）
        train_df = pd.read_csv(TRAIN_PATH)
        val_df = pd.read_csv(VAL_PATH)
        df = pd.concat([train_df, val_df], ignore_index=True)
        texts = df["text"].tolist()

        # ---- 词级 TF-IDF（jieba 分词 + unigram+bigram） ----
        from src.preprocess import tokenize
        word_texts = [" ".join(tokenize(t)) for t in texts]
        self._vec_word = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))
        self._vec_word.fit(word_texts)

        # ---- 字级 TF-IDF（原始文本 + 1/2/3-gram） ----
        self._vec_char = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, analyzer="char", ngram_range=(1, 3))
        self._vec_char.fit(texts)

        # 标记为组合向量器模式
        self.vectorizer = "xgboost_combined"

    # ========================================================================
    # 深度学习从零训练模型加载
    # ========================================================================

    def _load_dl(self, **kwargs):
        """加载 TextCNN / BiGRU-Attention 模型

        步骤：
          1. 从 word2idx 获取词表信息
          2. 重建模型架构（与训练时的结构一致）
          3. 加载训练好的 state_dict
          4. 设为 eval 模式（禁用 Dropout）
        """
        from src.config import EMBEDDING_DIM, MAX_SEQ_LEN
        word2idx = kwargs.get("word2idx")
        if word2idx is None:
            raise ValueError("DL models require word2idx")

        if self.model_type == "textcnn":
            from src.models.textcnn import TextCNN
            self.model = TextCNN(vocab_size=len(word2idx), embed_dim=EMBEDDING_DIM)
        else:
            from src.models.bigru_attn import BiGRUAttention
            self.model = BiGRUAttention(vocab_size=len(word2idx), embed_dim=EMBEDDING_DIM)

        # 加载权重到 CPU（避免 GPU→CPU 映射问题）
        self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        self.model.eval()  # 评估模式：禁用 Dropout 和 BatchNorm 训练行为
        self.word2idx = word2idx
        self.max_len = MAX_SEQ_LEN

    # ========================================================================
    # 预训练模型加载
    # ========================================================================

    def _load_pretrained(self, **kwargs):
        """加载 BERT / RoBERTa 微调模型

        直接使用 transformers 的 from_pretrained 加载完整模型（架构+权重+配置）
        和 tokenizer（词表+分词规则）。保存时使用 save_pretrained 完整保存，
        加载时只需传入保存目录路径即可。
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.eval()
        self.max_len = kwargs.get("max_len", 256)

    # ========================================================================
    # 统一推理接口
    # ========================================================================

    def predict(self, text: str) -> tuple[str, list[float]]:
        """对单条文本进行情感预测

        根据模型类型自动选择对应的预处理和推理流程：
          - 传统 ML:  jieba分词 → TF-IDF转换 → model.predict_proba
          - DL 从零:   jieba分词 → 索引序列 → model.forward → softmax
          - 预训练:   直接传入文本 → tokenizer → model.forward → softmax

        Args:
            text: 评论文本（单条）

        Returns:
            (predicted_label, class_probabilities):
              - label:  "差评" 或 "好评"
              - probs:  [neg_prob, pos_prob]，和为 1
        """
        # ---- 传统机器学习 ----
        if self.model_type in ("naive_bayes", "svm", "xgboost"):
            import jieba
            from src.preprocess import clean_text
            cleaned = clean_text(text)

            if self.model_type == "xgboost" and hasattr(self, "_vec_char"):
                # 调优模式 XGBoost：word + char 组合特征
                from scipy.sparse import hstack
                word_tokens = " ".join(jieba.cut(cleaned))
                w = self._vec_word.transform([word_tokens])       # 词级 TF-IDF
                c = self._vec_char.transform([cleaned])          # 字级 TF-IDF
                vec = hstack([w, c])                              # 水平拼接
            else:
                # 默认模式：jieba 分词 + 标准 TF-IDF
                tokens = " ".join(jieba.cut(cleaned))
                vec = self.vectorizer.transform([tokens])

            probs = self.model.predict_proba(vec)[0]
            probs = probs.tolist() if hasattr(probs, "tolist") else list(probs)

        # ---- 深度学习从零训练 ----
        elif self.model_type in ("textcnn", "bigru_attn"):
            from src.preprocess import tokenize
            # 分词 → 索引 → 截断 → 补齐
            tokens = tokenize(text)
            ids = [self.word2idx.get(t, 1) for t in tokens][:self.max_len]  # UNK=1
            attn = [1] * len(ids)        # attention_mask: 1=有效词元
            pad_len = self.max_len - len(ids)
            ids += [0] * pad_len          # PAD=0
            attn += [0] * pad_len

            with torch.no_grad():
                input_ids = torch.tensor([ids])
                attention_mask = torch.tensor([attn])
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1).squeeze().tolist()

        # ---- 预训练模型 ----
        else:  # bert / roberta
            from src.preprocess import clean_text
            cleaned = clean_text(text)
            # tokenizer 自动处理截断和补齐
            encoded = self.tokenizer(
                cleaned, truncation=True, padding="max_length",
                max_length=self.max_len, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=1).squeeze().tolist()

        # 取概率最大者作为预测标签
        pred_label = self.labels[probs.index(max(probs))]
        return pred_label, probs
