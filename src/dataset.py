import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from src.preprocess import tokenize


class TokenizedDataset(Dataset):
    """Pre-tokenized text index sequences for TextCNN/BiGRU (tokenizes once in __init__)"""

    def __init__(self, csv_path: str, word2idx: dict, max_len: int = 128):
        df = pd.read_csv(csv_path)
        self.max_len = max_len
        self.word2idx = word2idx
        self.pad_idx = word2idx.get("<PAD>", 0)
        self.unk_idx = word2idx.get("<UNK>", 1)

        # Pre-tokenize once to avoid re-running jieba.cut every epoch
        self.labels = df["label"].tolist()
        self.token_ids = []
        self.attention_masks = []
        for text in df["text"]:
            tokens = tokenize(text)
            ids = [word2idx.get(t, self.unk_idx) for t in tokens][:max_len]
            mask = [1] * len(ids)
            pad_len = max_len - len(ids)
            ids += [self.pad_idx] * pad_len
            mask += [0] * pad_len
            self.token_ids.append(ids)
            self.attention_masks.append(mask)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.token_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def create_data_loader(
    csv_path: str, word2idx: dict, batch_size: int = 64, max_len: int = 128, shuffle: bool = True
) -> DataLoader:
    dataset = TokenizedDataset(csv_path, word2idx, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_vocab_from_csv(csv_path: str, min_freq: int = 2, max_vocab: int = 30000) -> dict:
    """Build vocabulary from training data"""
    from collections import Counter
    df = pd.read_csv(csv_path)
    counter = Counter()
    total = len(df)
    for i, text in enumerate(df["text"], 1):
        counter.update(tokenize(text))
        if i % 10000 == 0:
            print(f"  Tokenizing... {i}/{total} ({100*i/total:.0f}%)")
    print(f"  Tokenizing done. Raw vocab size: {len(counter):,}")

    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, freq in counter.most_common(max_vocab):
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    print(f"  Vocab built: {len(vocab):,} words (min_freq={min_freq}, max_vocab={max_vocab})")
    return vocab


def build_embedding_matrix(word2idx: dict, wv_path: str = None, embed_dim: int = 300) -> np.ndarray:
    """Build embedding matrix from pretrained word vectors; random init if wv_path is None"""
    matrix = np.random.normal(scale=0.01, size=(len(word2idx), embed_dim)).astype(np.float32)
    matrix[0] = 0.0  # <PAD>

    if wv_path is None:
        print(f"No pretrained vectors provided, using random init (vocab={len(word2idx)}, dim={embed_dim})")
        return matrix

    from gensim.models import KeyedVectors
    import os
    size_mb = os.path.getsize(wv_path) / 1024**2
    print(f"Loading pretrained word vectors ({size_mb:.0f} MB), this may take a few minutes...")
    wv = KeyedVectors.load_word2vec_format(wv_path, binary=False)

    hit = 0
    for word, idx in word2idx.items():
        if word in wv:
            matrix[idx] = wv[word]
            hit += 1

    print(f"Vocab coverage: {hit}/{len(word2idx)} ({100 * hit / len(word2idx):.1f}%)")
    return matrix
