"""
DeepSeis - 2D CNN Temel Modeli
================================

Spektrogramı 2D görüntü olarak işleyen CNN modeli.
Basit ama etkili bir referans model olarak kullanılır.

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    """
    2D CNN ile sismik anomali sınıflandırıcı.

    Girdi: (batch, 1, freq_bins, time_steps)
    Çıktı: (batch, 2)

    Mimari:
        Conv2D → BN → ReLU → MaxPool (×3) → AdaptiveAvgPool → FC
    """

    def __init__(self, freq_bins: int = 129,
                 time_steps: int = 95,
                 num_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Blok 1: 1 → 32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Blok 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Blok 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_cnn(freq_bins: int = 129, time_steps: int = 95, **kwargs):
    """CNN modeli oluşturur."""
    return CNNClassifier(freq_bins=freq_bins, time_steps=time_steps, **kwargs)
