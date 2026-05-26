"""
DeepSeis - PyTorch Dataset Sınıfı
===================================

Spektrogram verilerini ve etiketleri PyTorch formatında sunar.

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class SeismicDataset(Dataset):
    """
    Sismik STFT spektrogramları için PyTorch Dataset.

    Parameters
    ----------
    X : np.ndarray
        Spektrogramlar, şekil: (N, freq_bins, time_steps)
    y : np.ndarray
        Etiketler, şekil: (N,)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y_np = np.asarray(y, dtype=np.int64)
        self.y = torch.LongTensor(self.y_np)

        # CNN için kanal boyutu ekle: (N, 1, F, T)
        self.add_channel_dim = self.X.ndim == 3

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.as_tensor(np.array(self.X[idx], copy=True), dtype=torch.float32)
        if self.add_channel_dim:
            x = x.unsqueeze(0)
        return x, self.y[idx]

    def get_class_weights(self) -> torch.Tensor:
        """
        Sınıf dengesizliğini telafi etmek için ağırlıklar hesaplar.
        """
        counts = np.bincount(self.y_np)
        weights = 1.0 / counts.astype(np.float32)
        weights = weights / weights.sum()
        return torch.FloatTensor(weights)

    def get_sampler(self) -> WeightedRandomSampler:
        """
        Dengeli örnekleme için WeightedRandomSampler döndürür.
        """
        counts = np.bincount(self.y_np)
        class_weights = 1.0 / counts.astype(np.float32)
        sample_weights = class_weights[self.y_np]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self),
            replacement=True
        )


def create_dataloaders(splits_dir: str,
                       batch_size: int = 64,
                       balanced: bool = True,
                       num_workers: int = 0) -> dict:
    """
    Train/Val/Test DataLoader'larını oluşturur.

    Parameters
    ----------
    splits_dir : str
        Bölünmüş veri dizini.
    batch_size : int
        Batch boyutu.
    balanced : bool
        Dengeli örnekleme kullanılsın mı.
    num_workers : int
        DataLoader worker sayısı.

    Returns
    -------
    dict
        DataLoader'lar ve Dataset'ler sözlüğü.
    """
    from pathlib import Path
    splits_path = Path(splits_dir)
    manifest = None
    manifest_path = splits_path / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    def resolve_x_path(split_name: str) -> Path:
        direct_path = splits_path / f"X_{split_name}.npy"
        if direct_path.exists():
            return direct_path
        if manifest and split_name in manifest.get("splits", {}):
            root = Path(__file__).resolve().parents[2]
            return root / manifest["splits"][split_name]["x_path"]
        return direct_path

    result = {}

    for split in ["train", "val", "test"]:
        X = np.load(str(resolve_x_path(split)), mmap_mode="r")
        y = np.load(str(splits_path / f"y_{split}.npy"))

        dataset = SeismicDataset(X, y)

        if split == "train" and balanced:
            sampler = dataset.get_sampler()
            loader = DataLoader(
                dataset, batch_size=batch_size,
                sampler=sampler, num_workers=num_workers,
                pin_memory=True
            )
        else:
            loader = DataLoader(
                dataset, batch_size=batch_size,
                shuffle=(split == "train" and not balanced),
                num_workers=num_workers, pin_memory=True
            )

        result[split] = {
            "dataset": dataset,
            "loader": loader,
            "size": len(dataset),
            "positive": int(y.sum()),
            "negative": len(y) - int(y.sum())
        }

    return result
