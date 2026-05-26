"""
DeepSeis - Sismik Transformer Modeli
======================================

Zaman serisi (spektrogram) analizi için uyarlanmış
Transformer tabanlı (Attention mekanizması olan) sınıflandırma modeli.

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Zaman serisindeki adımların sırasını modele öğretmek için
    kullanılan Konumsal Kodlama (Sinüs/Kosinüs dalgaları) katmanı.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # pe shape: (max_len, 1, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        
        # Buffer olarak kaydet (parametre değil ama state_dict içinde saklanır)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, time_steps, d_model)
        """
        # Girdiye konumsal kodlamayı ekle
        x = x + self.pe[:, :x.size(1), :]
        return x


class SeismicTransformer(nn.Module):
    """
    Spektrogram okuyabilen Transformer modeli.
    
    Girdi: (batch, 1, frekans_boyutu, zaman_adımı)
    """
    def __init__(self, 
                 freq_bins: int = 129, 
                 time_steps: int = 95,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_encoder_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.3,
                 num_classes: int = 2):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. Feature Embedding (Lineer Projeksiyon)
        # 129 frekansı d_model boyutuna izdüşümler
        self.embedding = nn.Linear(freq_bins, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=time_steps)
        self.dropout_pos = nn.Dropout(p=dropout)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # PyTorch'ta girdi batch ile başlasın diye
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # 4. Sınıflandırma Başı (Classification Head)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        x: (batch, 1, freq_bins, time_steps)
        """
        # Boyutları düzelt: (batch, time_steps, freq_bins) haline getir
        x = x.squeeze(1).permute(0, 2, 1) # (batch, 95, 129)
        
        # Embedding
        x = self.embedding(x) # (batch, 95, 128)
        
        # Positional Encoding (Transformer sıralamayı anlasın diye)
        # Ölçekleme faktörü eklemek (math.sqrt(d_model)) iyi bir pratik
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.dropout_pos(x)
        
        # Transformer Katmanları
        # (batch_first=True olduğu için direk (batch, seq, feature) verebiliriz)
        out = self.transformer_encoder(x) # (batch, 95, 128)
        
        # Global Average Pooling (Zaman ekseni boyunca ortalama al)
        # Böylece cümlenin tüm özetini tek bir vektöre indirgeriz
        out = out.mean(dim=1) # (batch, 128)
        
        # Sınıflandırma
        logits = self.classifier(out) # (batch, 2)
        
        return logits


def build_transformer(freq_bins: int = 129, time_steps: int = 95, **kwargs):
    """Transformer modeli oluşturur."""
    return SeismicTransformer(freq_bins=freq_bins, time_steps=time_steps, **kwargs)
