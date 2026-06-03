import re
import jieba


def clean_text(text: str) -> str:
    """Remove HTML tags, URLs, special chars; keep Chinese, English, digits"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^一-龥a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """jieba tokenization, returns word list"""
    cleaned = clean_text(text)
    return [w for w in jieba.cut(cleaned) if w.strip()]


def tokenize_for_pretrained(text: str) -> str:
    """For BERT/RoBERTa: clean only, no jieba (tokenizer handles it)"""
    return clean_text(text)
