"""
文本预处理模块
=============================================================================
提供文本清洗和分词的统一接口，被所有训练脚本和推理模块调用。

主要功能：
  - clean_text():     移除 HTML 标签、URL、特殊字符，保留中英文和数字
  - tokenize():       jieba 分词，用于传统 ML 和深度学习从零训练
  - tokenize_for_pretrained(): 仅清洗不分词，预训练模型使用自带 tokenizer

注意区分：
  - 传统 ML / TextCNN / BiGRU：需要先 jieba 分词，再构建词表或 TF-IDF
  - BERT / RoBERTa：直接传入原始清洗文本，由预训练 tokenizer 处理
"""

import re
import jieba


def clean_text(text: str) -> str:
    """清洗文本：去除 HTML、URL、特殊符号，保留中文、英文和数字

    处理步骤：
      1. 去除 HTML 标签（如 <br>, <div>）
      2. 去除 URL 链接（http/https）
      3. 移除非中文、非英文、非数字的所有字符
      4. 合并多余空白字符

    Args:
        text: 原始评论文本

    Returns:
        清洗后的干净文本，失败时返回空字符串
    """
    if not isinstance(text, str):
        return ""
    # 移除 HTML 标签：<任意内容>
    text = re.sub(r"<[^>]+>", " ", text)
    # 移除 URL：http:// 或 https:// 开头的链接
    text = re.sub(r"https?://\S+", " ", text)
    # 只保留中文（一-龥）、英文字母（a-zA-Z）、数字（0-9）
    text = re.sub(r"[^一-龥a-zA-Z0-9]", " ", text)
    # 合并多个空白符为单个空格，并去除首尾空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """使用 jieba 对清洗后的文本进行分词

    先调用 clean_text() 清洗，再用 jieba 分词，过滤掉空白词元。

    jieba 分词模式说明：
      默认使用精确模式（cut），适合文本分析场景。
      不会使用全模式或搜索引擎模式，因为精确模式已经足够。

    示例：
      tokenize("产品质量很好") → ["产品", "质量", "很", "好"]

    Args:
        text: 原始评论文本

    Returns:
        分词后的词语列表
    """
    cleaned = clean_text(text)
    return [w for w in jieba.cut(cleaned) if w.strip()]


def tokenize_for_pretrained(text: str) -> str:
    """为 BERT/RoBERTa 准备输入：仅清洗，保留为原始字符串

    预训练模型自带的 tokenizer（如 BertTokenizer）会进行 WordPiece 或 BPE
    子词切分，不需要也不应该用 jieba 预先分词。预先分词反而会破坏子词信息。

    示例：
      tokenize_for_pretrained("产品质量很好！<br>") → "产品质量很好"

    Args:
        text: 原始评论文本

    Returns:
        清洗后的文本字符串（不做分词）
    """
    return clean_text(text)
