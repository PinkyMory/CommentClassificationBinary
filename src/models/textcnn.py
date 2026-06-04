"""
TextCNN 文本分类模型
=============================================================================
基于论文 "Convolutional Neural Networks for Sentence Classification" (Kim, 2014)。

核心思想：
  使用多个不同尺寸的卷积核（3/4/5-gram）在词向量序列上做一维卷积，
  每个卷积核提取不同粒度的局部 n-gram 特征，然后通过最大池化保留最显著特征，
  最后拼接所有卷积核的输出并送入全连接层分类。

为什么用多尺寸卷积核？
  - 3-gram: 捕捉短语级特征（"很不错"、"太差了"）
  - 4-gram: 捕捉短句级特征（"质量非常好"）
  - 5-gram: 捕捉更长的局部模式
  多粒度组合能同时利用不同层次的语言信息。

模型结构：
  Embedding → [Conv2d(3,4,5-gram)] → ReLU → MaxPool → Concat → Dropout → FC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """TextCNN: 多尺寸卷积核（3/4/5-gram）文本分类器

    输入形状：
      - input_ids:       (batch_size, seq_len)
      - attention_mask:  (batch_size, seq_len) —— 当前版本未直接使用，
                          因为 padding 位置在 MaxPool 后自然被压缩

    输出形状：
      - logits: (batch_size, num_classes)

    参数规模（以 vocab=30000, embed_dim=300, num_filters=100 为例）：
      - Embedding:  30000 × 300 = 9M 参数
      - 3 组卷积:   3 × (3+4+5) × 300 × 100 ≈ 1M 参数
      - FC:         300 × 2 = 600 参数
    """

    def __init__(self, vocab_size, embed_dim=300, num_filters=100,
                 filter_sizes=(3, 4, 5), num_classes=2, dropout=0.5,
                 pretrained_embeddings=None, freeze_embeddings=False):
        """
        Args:
            vocab_size:            词表大小
            embed_dim:             词向量维度（默认 300）
            num_filters:           每种卷积核的数量（默认 100）
            filter_sizes:          卷积核的 n-gram 尺寸元组
            num_classes:           分类类别数（二分类 = 2）
            dropout:               Dropout 概率
            pretrained_embeddings: 预训练词向量矩阵 [vocab_size, embed_dim]
            freeze_embeddings:     是否冻结 Embedding 层（不参与训练）
        """
        super().__init__()

        # ---- Embedding 层 ----
        # padding_idx=0 确保 <PAD> 位置的梯度始终为 0，不被更新
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            # 用预训练词向量初始化 Embedding 权重
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            # 冻结 Embedding 层，减少可训练参数，防止过拟合
            self.embedding.weight.requires_grad = False

        # ---- 多尺寸卷积层 ----
        # 每个 Conv2d 的 kernel_size 为 (filter_size, embed_dim)
        # 即卷积核在词向量维度上跨度 = embed_dim（一次性覆盖整个词向量），
        # 在序列维度上跨度 = filter_size（覆盖 n 个相邻词）
        # 这等价于沿着序列方向的一维卷积，但用 Conv2d 实现
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,                    # 单通道输入（词向量序列）
                out_channels=num_filters,         # 每种尺寸输出 num_filters 个特征图
                kernel_size=(fs, embed_dim)       # (n-gram 宽度, 词向量整个维度)
            )
            for fs in filter_sizes
        ])

        # ---- 分类头 ----
        self.dropout = nn.Dropout(dropout)
        # 输入维度 = num_filters × len(filter_sizes)
        # 例如 100 × 3 = 300 维
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids:      (batch_size, seq_len) 词索引序列
            attention_mask: (batch_size, seq_len) 当前未直接使用

        Returns:
            logits: (batch_size, num_classes)
        """
        # 1. Embedding: (batch, seq_len) → (batch, seq_len, embed_dim)
        emb = self.embedding(input_ids)

        # 2. 增加通道维度: (batch, 1, seq_len, embed_dim)
        #    Conv2d 期望 (N, C_in, H, W) 格式
        emb = emb.unsqueeze(1)

        pooled = []
        for conv in self.convs:
            # 3. 卷积: (batch, 1, seq_len, embed_dim) → (batch, num_filters, L_out, 1)
            c = F.relu(conv(emb))
            # 4. 压缩最后一维: (batch, num_filters, L_out)
            c = c.squeeze(3)
            # 5. Max-over-time Pooling: 对每个 filter 取整个序列的最大值
            #    (batch, num_filters, L_out) → (batch, num_filters, 1)
            c = F.max_pool1d(c, c.size(2))
            # 6. 压缩: (batch, num_filters)
            pooled.append(c.squeeze(2))

        # 7. 拼接所有卷积核的输出: (batch, num_filters × 3)
        out = torch.cat(pooled, dim=1)
        # 8. Dropout 正则化
        out = self.dropout(out)
        # 9. 全连接分类
        return self.fc(out)
