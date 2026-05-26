"""
DeepSeis - Bidirectional GRU Modeli
=====================================

LSTM ile benzer yapıda ancak daha hafif bir alternatif.
GRU, LSTM'den daha az parametreye sahiptir.

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import torch
import torch.nn as nn


class BiGRUClassifier(nn.Module):
    """
    Bidirectional GRU ile sismik anomali sınıflandırıcı.

    Girdi: (batch, 1, freq_bins, time_steps)
    Çıktı: (batch, 2)
    """

    def __init__(self, freq_bins: int = 129,
                 time_steps: int = 95,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 num_classes: int = 2):
        super().__init__()

        self.gru = nn.GRU(
            input_size=freq_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Sınıflandırıcı
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # (batch, 1, freq, time) → (batch, time, freq)
        x = x.squeeze(1).permute(0, 2, 1)

        gru_out, _ = self.gru(x)

        # Attention
        attn_weights = self.attention(gru_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (gru_out * attn_weights).sum(dim=1)

        out = self.classifier(context)
        return out


def build_gru(freq_bins: int = 129, time_steps: int = 95, **kwargs):
    """GRU modeli oluşturur."""
    return BiGRUClassifier(freq_bins=freq_bins, time_steps=time_steps, **kwargs)
