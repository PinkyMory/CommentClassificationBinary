"""Unified model loading and inference interface (binary)"""
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


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
        self.model = pickle.load(open(self.model_path, "rb"))
        self.vectorizer = kwargs.get("vectorizer")
        if self.vectorizer is None:
            raise ValueError("Traditional models require vectorizer (TfidfVectorizer)")

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
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
            tokens = " ".join(jieba.cut(cleaned))
            vec = self.vectorizer.transform([tokens])
            probs = self.model.predict_proba(vec)[0]

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
