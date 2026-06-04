"""
数据集与词表构建模块
=============================================================================
为深度学习从零训练（TextCNN / BiGRU-Attention）提供：
  - TokenizedDataset:      预分词的数据集类，将文本转为索引序列
  - build_vocab_from_csv:  从训练数据构建词表（word → index 映射）
  - build_embedding_matrix: 构建词向量矩阵，支持加载预训练词向量
  - create_data_loader:    工厂函数，创建 DataLoader

关键设计：
  TokenizedDataset 在 __init__ 中一次性完成所有文本的 jieba 分词和索引
  转换，而非在 __getitem__ 中逐个分词。这样避免了每个 epoch 都重复运行
  jieba.cut() 的开销，显著提升训练速度。
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from src.preprocess import tokenize


class TokenizedDataset(Dataset):
    """预分词文本索引数据集

    在初始化阶段一次性完成以下操作：
      1. 读取 CSV 文件
      2. 对每条文本进行 jieba 分词
      3. 将词语转换为索引序列
      4. 处理截断和补齐
      5. 生成 attention_mask

    这样 __getitem__ 只需做张量创建，无需重复分词。

    Attention: 不要将 jieba.cut() 移到 __getitem__ 中——
    那样每个 epoch 都会重新分词，训练速度会大幅下降。
    """

    def __init__(self, csv_path: str, word2idx: dict, max_len: int = 128):
        """
        Args:
            csv_path: 数据 CSV 文件路径，需包含 "text" 和 "label" 两列
            word2idx: 词到索引的映射字典，PAD→0, UNK→1
            max_len:  最大序列长度，超出截断、不足补齐
        """
        df = pd.read_csv(csv_path)
        self.max_len = max_len
        self.word2idx = word2idx
        # <PAD> 填充索引，通常为 0
        self.pad_idx = word2idx.get("<PAD>", 0)
        # <UNK> 未知词索引，通常为 1
        self.unk_idx = word2idx.get("<UNK>", 1)

        self.labels = df["label"].tolist()
        self.token_ids = []
        self.attention_masks = []

        # 逐条进行分词 → 索引转换 → 截断 → 补齐
        for text in df["text"]:
            tokens = tokenize(text)
            # 将词语转为索引，超出词表范围的词用 UNK 代替
            ids = [word2idx.get(t, self.unk_idx) for t in tokens][:max_len]
            # attention_mask: 1表示真实词元，0表示PAD位置
            mask = [1] * len(ids)
            # 不足 max_len 的部分用 PAD 补齐
            pad_len = max_len - len(ids)
            ids += [self.pad_idx] * pad_len
            mask += [0] * pad_len
            self.token_ids.append(ids)
            self.attention_masks.append(mask)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """返回一个样本的字典：input_ids, attention_mask, label"""
        return {
            "input_ids": torch.tensor(self.token_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def create_data_loader(
    csv_path: str, word2idx: dict, batch_size: int = 64,
    max_len: int = 128, shuffle: bool = True
) -> DataLoader:
    """便捷工厂函数：从 CSV 路径直接创建 DataLoader

    Args:
        csv_path:   数据 CSV 文件路径
        word2idx:   词表（{word: index}）
        batch_size: 批大小
        max_len:    序列最大长度
        shuffle:    是否打乱数据（训练时 True，验证/测试时 False）

    Returns:
        配置好的 PyTorch DataLoader
    """
    dataset = TokenizedDataset(csv_path, word2idx, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_vocab_from_csv(csv_path: str, min_freq: int = 2, max_vocab: int = 30000) -> dict:
    """从训练集 CSV 构建词表（word → index 映射）

    构建流程：
      1. 遍历所有文本，统计词频
      2. 保留词频 >= min_freq 的词语（过滤低频噪音词）
      3. 按词频从高到低排序，最多保留 max_vocab 个词
      4. 特殊标记：<PAD>=0 用于补齐，<UNK>=1 用于未知词

    参数选择建议：
      - min_freq=2: 过滤只出现 1 次的词（噪音/错别字），减少词表噪音
      - max_vocab=30000: 足够覆盖大多数中文电商评论的常用词汇

    Args:
        csv_path:  训练集 CSV 路径
        min_freq:  词语最小出现次数，低于此频率的词语被丢弃
        max_vocab: 词表最大容量（不含 <PAD> 和 <UNK>）

    Returns:
        {word: index} 映射字典
    """
    from collections import Counter
    df = pd.read_csv(csv_path)
    counter = Counter()
    total = len(df)
    # 逐条分词并统计词频
    for i, text in enumerate(df["text"], 1):
        counter.update(tokenize(text))
        if i % 10000 == 0:
            print(f"  Tokenizing... {i}/{total} ({100*i/total:.0f}%)")
    print(f"  Tokenizing done. Raw vocab size: {len(counter):,}")

    # 构建词表，保留 <PAD> 和 <UNK> 两个特殊位置
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, freq in counter.most_common(max_vocab):
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    print(f"  Vocab built: {len(vocab):,} words (min_freq={min_freq}, max_vocab={max_vocab})")
    return vocab


def build_embedding_matrix(
    word2idx: dict, wv_path: str = None, embed_dim: int = 300
) -> np.ndarray:
    """构建词向量矩阵（Embedding 层的初始权重）

    工作流程：
      1. 用正态分布随机初始化 [vocab_size, embed_dim] 矩阵
      2. 将 <PAD> 行设为全零（填充位置不应有梯度）
      3. 如果提供了预训练词向量路径，则加载并用预训练向量覆盖匹配的词语
      4. 未匹配的词语保持随机初始化

    这样即使没有预训练词向量也能正常训练，有预训练向量则效果更好。

    Args:
        word2idx:   词表映射字典
        wv_path:    预训练词向量文件路径（word2vec 格式），None 则全部随机初始化
        embed_dim:  词向量维度，默认 300

    Returns:
        numpy 数组，形状 [vocab_size, embed_dim]，dtype=float32
    """
    # 随机初始化：均值 0，标准差 0.01 的正态分布
    matrix = np.random.normal(scale=0.01, size=(len(word2idx), embed_dim)).astype(np.float32)
    # <PAD> 的第 0 行全部置零，保证填充位置在 Embedding 查找后输出零向量
    matrix[0] = 0.0

    if wv_path is None:
        print(f"No pretrained vectors provided, using random init (vocab={len(word2idx)}, dim={embed_dim})")
        return matrix

    # 加载预训练词向量（文本格式的 word2vec 文件）
    from gensim.models import KeyedVectors
    import os
    size_mb = os.path.getsize(wv_path) / 1024**2
    print(f"Loading pretrained word vectors ({size_mb:.0f} MB), this may take a few minutes...")
    # binary=False 表示文本格式（每行一个词 + 空格分隔的向量）
    wv = KeyedVectors.load_word2vec_format(wv_path, binary=False)

    # 用预训练向量覆盖匹配的词语
    hit = 0
    for word, idx in word2idx.items():
        if word in wv:
            matrix[idx] = wv[word]
            hit += 1

    print(f"Vocab coverage: {hit}/{len(word2idx)} ({100 * hit / len(word2idx):.1f}%)")
    return matrix
