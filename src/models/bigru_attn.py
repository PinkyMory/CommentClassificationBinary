import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Additive attention (Bahdanau-style)"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, gru_output, mask=None):
        # gru_output: (batch, seq_len, hidden_dim * 2)
        scores = self.v(torch.tanh(self.W(gru_output)))  # (batch, seq_len, 1)
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.sum(attn_weights * gru_output, dim=1)
        return context, attn_weights.squeeze(-1)


class BiGRUAttention(nn.Module):
    """BiGRU + Attention text classifier"""
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=128, num_layers=1,
                 num_classes=2, dropout=0.5, pretrained_embeddings=None, freeze_embeddings=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = Attention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids)
        gru_out, _ = self.gru(emb)          # (batch, seq_len, hidden_dim*2)
        context, _ = self.attention(gru_out, attention_mask)  # (batch, hidden_dim*2)
        out = self.dropout(context)
        return self.fc(out)
