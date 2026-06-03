"""Unified model loading and inference interface (binary)"""
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentPredictor:
    """Loads any trained model, provides a unified predict interface"""

    def __init__(self, model_type: str, model_path: str, **kwargs):
        """
        Args:
            model_type: "naive_bayes"|"svm"|"xgboost"|"textcnn"|"bigru_attn"|"bert"|"roberta"
            model_path: path to model weights
        """
        self.model_type = model_type
        self.model_path = model_path
        self.labels = ["差评", "好评"]

        if model_type in ("naive_bayes", "svm", "xgboost"):
            self._load_traditional(**kwargs)
        elif model_type in ("textcnn", "bigru_attn"):
            self._load_dl(**kwargs)
        elif model_type in ("bert", "roberta"):
            self._load_pretrained(**kwargs)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def _load_traditional(self, **kwargs):
        if self.model_type == "xgboost":
            from xgboost import XGBClassifier
            self.model = XGBClassifier()
            self.model.load_model(self.model_path)
            self.vectorizer = kwargs.get("vectorizer")
            if self.vectorizer is None:
                # Rebuild word+char TF-IDF for tuned XGBoost (10000-dim)
                self._build_xgboost_vectorizers()
        else:
            self.model = pickle.load(open(self.model_path, "rb"))
            self.vectorizer = kwargs.get("vectorizer")
            if self.vectorizer is None:
                raise ValueError("Traditional models require vectorizer (TfidfVectorizer)")

    def _build_xgboost_vectorizers(self):
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse import hstack
        from src.config import TRAIN_PATH, VAL_PATH, TFIDF_MAX_FEATURES

        train_df = pd.read_csv(TRAIN_PATH)
        val_df = pd.read_csv(VAL_PATH)
        df = pd.concat([train_df, val_df], ignore_index=True)
        texts = df["text"].tolist()

        # Word-level (jieba)
        from src.preprocess import tokenize
        word_texts = [" ".join(tokenize(t)) for t in texts]
        self._vec_word = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))
        self._vec_word.fit(word_texts)

        # Char-level
        self._vec_char = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, analyzer="char", ngram_range=(1, 3))
        self._vec_char.fit(texts)

        # Store as a combined pipeline for predict
        self.vectorizer = "xgboost_combined"  # marker for predict method

    def _load_dl(self, **kwargs):
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
        self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        self.model.eval()
        self.word2idx = word2idx
        self.max_len = MAX_SEQ_LEN

    def _load_pretrained(self, **kwargs):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.eval()
        self.max_len = kwargs.get("max_len", 256)

    def predict(self, text: str) -> tuple[str, list[float]]:
        """Returns (predicted_label, class_probabilities)"""
        if self.model_type in ("naive_bayes", "svm", "xgboost"):
            import jieba
            from src.preprocess import clean_text
            cleaned = clean_text(text)
            if self.model_type == "xgboost" and hasattr(self, "_vec_char"):
                # Tuned XGBoost: word + char combined features
                from scipy.sparse import hstack
                word_tokens = " ".join(jieba.cut(cleaned))
                w = self._vec_word.transform([word_tokens])
                c = self._vec_char.transform([cleaned])
                vec = hstack([w, c])
            else:
                tokens = " ".join(jieba.cut(cleaned))
                vec = self.vectorizer.transform([tokens])
            probs = self.model.predict_proba(vec)[0]
            probs = probs.tolist() if hasattr(probs, "tolist") else list(probs)

        elif self.model_type in ("textcnn", "bigru_attn"):
            from src.preprocess import tokenize
            tokens = tokenize(text)
            ids = [self.word2idx.get(t, 1) for t in tokens][:self.max_len]
            attn = [1] * len(ids)
            pad_len = self.max_len - len(ids)
            ids += [0] * pad_len
            attn += [0] * pad_len
            with torch.no_grad():
                input_ids = torch.tensor([ids])
                attention_mask = torch.tensor([attn])
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1).squeeze().tolist()

        else:  # bert / roberta
            from src.preprocess import clean_text
            cleaned = clean_text(text)
            encoded = self.tokenizer(
                cleaned, truncation=True, padding="max_length",
                max_length=self.max_len, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=1).squeeze().tolist()

        pred_label = self.labels[probs.index(max(probs))]
        return pred_label, probs
