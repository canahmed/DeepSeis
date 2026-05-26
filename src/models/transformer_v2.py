"""
DeepSeis - Sismik Transformer V2 (Hibrit CNN-Transformer)
==========================================================

Spektrogramdaki yerel frekans desenlerini CNN ile çıkarıp,
uzun vadeli zamansal bağımlılıkları Transformer ile modelleyen
gelişmiş hibrit mimari.

Focal Loss ile aşırı sınıf dengesizliğine dayanıklıdır.

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# FOCAL LOSS — Nadir sınıfları kaçırınca ağır ceza veren kayıp
# ============================================================
class FocalLoss(nn.Module):
    """
    Lin et al. (2017) - Focal Loss for Dense Object Detection
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    gamma > 0 ise kolay örneklerin ağırlığını düşürür,
    modeli zor (nadir anomali) örneklere odaklanmaya zorlar.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Alpha weighting
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=logits.device),
            torch.tensor(1.0 - self.alpha, device=logits.device)
        )
        
        focal_loss = alpha_t * (1.0 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ============================================================
# ATTENTION POOLING — Önemli zaman adımlarına odaklanan havuzlama
# ============================================================
class AttentionPooling(nn.Module):
    """
    Global Average Pooling yerine, her zaman adımının
    önem ağırlığını öğrenen dikkat tabanlı havuzlama.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        # Her zaman adımının önem skorunu hesapla
        attn_weights = self.attention(x)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Ağırlıklı toplam
        out = (x * attn_weights).sum(dim=1)  # (batch, d_model)
        return out


# ============================================================
# CNN FEATURE EXTRACTOR — Yerel frekans deseni çıkarıcı
# ============================================================
class CNNFeatureExtractor(nn.Module):
    """
    Spektrogramdaki yerel frekans ve zaman desenlerini
    2D konvolüsyon ile çıkarır. Transformer'a temiz özellik verir.
    """
    def __init__(self, out_channels: int = 64, target_freq: int = 16):
        super().__init__()
        self.target_freq = target_freq
        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            
            # Block 2
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        # Frekans boyutunu sabit bir boyuta indir
        self.freq_pool = nn.AdaptiveAvgPool2d((target_freq, None))
    
    def forward(self, x):
        """
        x: (batch, 1, 129, 95)
        return: (batch, time_steps, out_channels * target_freq)
        """
        out = self.conv_block(x)  # (batch, 64, 129, 95)
        # Frekans boyutunu sabitle
        b, c, f, t = out.shape
        out = self.freq_pool(out)  # (batch, 64, 16, 95)
        b, c, f, t = out.shape
        out = out.permute(0, 3, 1, 2).reshape(b, t, c * f)  # (batch, 95, 1024)
        return out


# ============================================================
# POSITIONAL ENCODING — Sinüzoidal konum bilgisi
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================
# SEISMIC TRANSFORMER V2 — Hibrit CNN-Transformer
# ============================================================
class SeismicTransformerV2(nn.Module):
    """
    Geliştirilmiş Hibrit CNN-Transformer modeli.
    
    Akış:
        Spektrogram → CNN (yerel desen çıkarımı)
                     → Lineer Projeksiyon
                     → Positional Encoding
                     → Transformer Encoder (global zamansal bağımlılıklar)
                     → Attention Pooling
                     → Sınıflandırma (MLP)
    """
    def __init__(self,
                 freq_bins: int = 129,
                 time_steps: int = 95,
                 cnn_channels: int = 64,
                 target_freq: int = 16,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.2,
                 num_classes: int = 2):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. CNN Feature Extractor
        self.cnn = CNNFeatureExtractor(out_channels=cnn_channels, target_freq=target_freq)
        
        # CNN ciკis boyutu: cnn_channels * target_freq
        cnn_feature_dim = cnn_channels * target_freq  # 64 * 16 = 1024
        
        # 2. Lineer projeksiyon (CNN ciktisini d_model boyutuna indir)
        self.projection = nn.Sequential(
            nn.Linear(cnn_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 3. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=time_steps, dropout=dropout)
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN (daha stabil egitim)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # 5. Attention Pooling
        self.pool = AttentionPooling(d_model)
        
        # 6. Sınıflandırma Başı
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )
        
        # Parametre başlatma
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        """
        x: (batch, 1, freq_bins, time_steps) = (batch, 1, 129, 95)
        """
        # CNN ile yerel frekans desenleri çıkar
        features = self.cnn(x)  # (batch, 95, 2048)
        
        # d_model boyutuna projeksiyon
        features = self.projection(features)  # (batch, 95, 256)
        
        # Positional encoding
        features = features * math.sqrt(self.d_model)
        features = self.pos_encoder(features)
        
        # Transformer encoder
        encoded = self.transformer_encoder(features)  # (batch, 95, 256)
        
        # Attention pooling
        pooled = self.pool(encoded)  # (batch, 256)
        
        # Sınıflandırma
        logits = self.classifier(pooled)  # (batch, 2)
        
        return logits


def build_transformer_v2(freq_bins: int = 129, time_steps: int = 95, **kwargs):
    """SeismicTransformer V2 modeli oluşturur."""
    return SeismicTransformerV2(freq_bins=freq_bins, time_steps=time_steps, **kwargs)
