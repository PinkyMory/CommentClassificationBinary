"""
BiGRU + Attention 文本分类模型
=============================================================================
结合双向 GRU 和注意力机制的情感分类模型。

为什么用 GRU 而非 LSTM？
  - GRU 参数更少（2 个门控 vs LSTM 的 3 个门控），训练更快
  - 在短文本分类任务上，GRU 和 LSTM 性能相当
  - 更少的参数也意味着更低的过拟合风险

为什么用双向？
  - 单向只能利用上文信息，双向能同时利用上下文
  - 对情感分析来说，"不"后面的词（下文）对语义判断同样关键

为什么用注意力机制？
  - GRU 的最终隐藏状态会"遗忘"序列早期的信息（长距离依赖问题）
  - 注意力机制通过对所有时间步的隐藏状态加权求和，直接"关注"关键位置
  - 例如："虽然包装一般，但产品质量非常好" → 注意力应集中在"非常好"

模型结构：
  Embedding → BiGRU → Attention → Dropout → FC

注意力类型：加法注意力（Bahdanau-style / Additive Attention）
  score(h_t) = v^T · tanh(W · h_t)
  其中 h_t 是 BiGRU 在时间步 t 的隐藏状态（concat 了前向和后向）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """加法注意力层（Bahdanau Attention）

    计算过程：
      1. 对每个时间步的 GRU 输出做线性变换：W · h_t
      2. 通过 tanh 激活
      3. 用可学习向量 v 计算得分：score = v^T · tanh(W · h_t)
      4. 对得分做 softmax 得到注意力权重
      5. 加权求和得到上下文向量

    为什么选加法注意力而非点积注意力？
      加法注意力（Bahdanau）在小维度下表现更稳定，不要求 Q 和 K 维度相同，
      是文本分类任务中最常用的注意力形式。
    """

    def __init__(self, hidden_dim):
        """
        Args:
            hidden_dim: GRU 隐藏层维度（双向拼接后 = hidden_dim * 2）
        """
        super().__init__()
        # W: 将 GRU 输出映射到相同维度的隐藏空间
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # v: 将隐藏表示映射为一个标量得分
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, gru_output, mask=None):
        """
        Args:
            gru_output: BiGRU 所有时间步的输出，形状 (batch, seq_len, hidden_dim*2)
            mask:       attention_mask，形状 (batch, seq_len)，1=有效词元，0=PAD

        Returns:
            context:      上下文向量，形状 (batch, hidden_dim*2)
            attn_weights: 注意力权重，形状 (batch, seq_len)
        """
        # 计算每个时间步的注意力得分
        # gru_output:        (batch, seq_len, hidden_dim)
        # tanh(W(gru)):      (batch, seq_len, hidden_dim)
        # scores:            (batch, seq_len, 1)
        scores = self.v(torch.tanh(self.W(gru_output)))

        # 对 PAD 位置施加极大的负值，使其 softmax 后权重接近 0
        if mask is not None:
            mask = mask.unsqueeze(-1).float()       # (batch, seq_len, 1)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax 归一化得到注意力权重
        attn_weights = F.softmax(scores, dim=1)      # (batch, seq_len, 1)

        # 加权求和：每个时间步的 GRU 输出按其注意力权重加权
        context = torch.sum(attn_weights * gru_output, dim=1)  # (batch, hidden_dim)

        return context, attn_weights.squeeze(-1)


class BiGRUAttention(nn.Module):
    """BiGRU + Attention 文本分类器

    输入形状：
      - input_ids:       (batch_size, seq_len)
      - attention_mask:  (batch_size, seq_len)

    输出形状：
      - logits: (batch_size, num_classes)

    参数规模（以 vocab=30000, embed_dim=300, hidden_dim=128 为例）：
      - Embedding: 30000 × 300 = 9M 参数
      - BiGRU:     2 × 3 × (300×128 + 128² + 128) ≈ 350K 参数
      - Attention: (256×256 + 256×1) ≈ 66K 参数
      - FC:        256 × 2 = 512 参数
    """

    def __init__(self, vocab_size, embed_dim=300, hidden_dim=128, num_layers=1,
                 num_classes=2, dropout=0.5, pretrained_embeddings=None,
                 freeze_embeddings=False):
        """
        Args:
            vocab_size:            词表大小
            embed_dim:             词向量维度（默认 300）
            hidden_dim:            GRU 隐藏层维度（默认 128）
            num_layers:            GRU 层数（默认 1，多层可能在小数据集上过拟合）
            num_classes:           分类类别数
            dropout:               Dropout 概率
            pretrained_embeddings: 预训练词向量矩阵
            freeze_embeddings:     是否冻结 Embedding 层
        """
        super().__init__()
        # ---- Embedding 层 ----
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # ---- 双向 GRU 层 ----
        # bidirectional=True: 输出维度自动 ×2（前向 + 后向拼接）
        # batch_first=True: 输入输出格式为 (batch, seq, feature)，更直观
        # 多层 GRU 间使用 dropout（仅当 num_layers > 1 时生效）
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # ---- 注意力层 ----
        # 输入维度 = hidden_dim * 2（因为双向）
        self.attention = Attention(hidden_dim * 2)

        # ---- 分类头 ----
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids:      (batch, seq_len) 词索引序列
            attention_mask: (batch, seq_len) 1=有效词元，0=PAD

        Returns:
            logits: (batch, num_classes)
        """
        # 1. Embedding: (batch, seq_len) → (batch, seq_len, embed_dim)
        emb = self.embedding(input_ids)

        # 2. BiGRU: (batch, seq_len, embed_dim) → (batch, seq_len, hidden_dim*2)
        #    gru_out 包含所有时间步的输出（用于后续注意力计算）
        #    h_n 是最终隐藏状态（本模型不使用）
        gru_out, _ = self.gru(emb)

        # 3. Attention: 对 BiGRU 输出加权求和
        #    attention_mask 传给 Attention 层以屏蔽 PAD 位置的注意力
        context, _ = self.attention(gru_out, attention_mask)

        # 4. Dropout + 全连接分类
        out = self.dropout(context)
        return self.fc(out)
