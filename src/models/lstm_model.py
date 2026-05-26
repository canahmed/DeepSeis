"""
DeepSeis - Bidirectional LSTM Modeli
======================================

Spektrogram verisi üzerinde ikili sınıflandırma yapan
çift yönlü LSTM modeli. Temel karşılaştırma modeli olarak kullanılır.

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM ile sismik anomali sınıflandırıcı.

    Girdi: (batch, 1, freq_bins, time_steps)
    Çıktı: (batch, 2) — [normal, anomali] logits

    Spektrogramı zaman adımları boyunca LSTM'e besler:
        - Her zaman adımında frekans çubukları özellik vektörü olur
        - LSTM bu sıralı yapıyı öğrenir
    """

    def __init__(self, freq_bins: int = 129,
                 time_steps: int = 95,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 num_classes: int = 2):
        super().__init__()

        self.freq_bins = freq_bins
        self.time_steps = time_steps
        self.hidden_size = hidden_size

        # LSTM: giriş = frekans çubukları, sıra = zaman adımları
        self.lstm = nn.LSTM(
            input_size=freq_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mekanizması (basit)
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
        # x: (batch, 1, freq_bins, time_steps)
        # → (batch, time_steps, freq_bins)
        x = x.squeeze(1).permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, time_steps, hidden*2)

        # Attention
        attn_weights = self.attention(lstm_out)     # (batch, time_steps, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*2)

        # Sınıflandırma
        out = self.classifier(context)
        return out


def build_lstm(freq_bins: int = 129, time_steps: int = 95, **kwargs):
    """LSTM modeli oluşturur."""
    return BiLSTMClassifier(freq_bins=freq_bins, time_steps=time_steps, **kwargs)
