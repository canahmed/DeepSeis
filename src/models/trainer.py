"""
DeepSeis - Ortak Eğitim Döngüsü (Trainer)
============================================

Tüm modeller için ortak train/eval/predict altyapısı.
Early stopping, checkpoint kaydetme ve metrik takibi desteği.

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

logger = logging.getLogger("DeepSeis.Trainer")


class Trainer:
    """
    Ortak model eğitim sınıfı.

    Parameters
    ----------
    model : nn.Module
        PyTorch modeli.
    device : str
        "cuda" veya "cpu".
    checkpoint_dir : str
        Model kayıt dizini.
    """

    def __init__(self, model: nn.Module,
                 device: str = "auto",
                 checkpoint_dir: str = "artifacts/checkpoints"):

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_f1": [], "val_f1": [],
            "lr": []
        }

        logger.info(f"Cihaz: {self.device}")
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parametreleri: {total_params:,}")

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 50,
              lr: float = 1e-3,
              weight_decay: float = 1e-4,
              class_weights: torch.Tensor = None,
              patience: int = 10,
              model_name: str = "model",
              custom_criterion=None,
              use_cosine_schedule: bool = False) -> Dict:
        """
        Model eğitimi.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader
        epochs : int
        lr : float
        weight_decay : float
        class_weights : Tensor, optional
            Sınıf ağırlıkları (dengesizlik telafisi).
        patience : int
            Early stopping sabrı.
        model_name : str
            Kayıt dosya adı.

        Returns
        -------
        dict
            Eğitim geçmişi.
        """
        # Loss ve optimizer
        if custom_criterion is not None:
            criterion = custom_criterion
            logger.info(f"Ozel kayip fonksiyonu: {type(criterion).__name__}")
        elif class_weights is not None:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(self.device)
            )
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        if use_cosine_schedule:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
            logger.info("Cosine Annealing Warm Restarts scheduler aktif.")
        else:
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )

        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        logger.info(f"\nEğitim başlıyor: {epochs} epoch, lr={lr}")
        logger.info(f"{'='*70}")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # --- TRAIN ---
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_tp, train_fp, train_fn = 0, 0, 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)
                preds = outputs.argmax(dim=1)
                train_correct += (preds == y_batch).sum().item()
                train_total += y_batch.size(0)

                # F1 bileşenleri
                train_tp += ((preds == 1) & (y_batch == 1)).sum().item()
                train_fp += ((preds == 1) & (y_batch == 0)).sum().item()
                train_fn += ((preds == 0) & (y_batch == 1)).sum().item()

            train_loss /= train_total
            train_acc = train_correct / train_total
            train_prec = train_tp / (train_tp + train_fp + 1e-8)
            train_rec = train_tp / (train_tp + train_fn + 1e-8)
            train_f1 = 2 * train_prec * train_rec / (train_prec + train_rec + 1e-8)

            # --- VALIDATION ---
            val_loss, val_acc, val_f1 = self._evaluate(val_loader, criterion)

            # Scheduler
            if use_cosine_schedule:
                scheduler.step(epoch)
            else:
                scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # History
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["train_f1"].append(train_f1)
            self.history["val_f1"].append(val_f1)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
                f"LR: {current_lr:.2e} | {elapsed:.1f}s"
            )

            # Early stopping & checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint(model_name, epoch, val_loss, val_f1)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"\nEarly stopping! En iyi epoch: {best_epoch} "
                             f"(val_loss: {best_val_loss:.4f})")
                break

        # En iyi modeli yükle
        self._load_checkpoint(model_name)
        logger.info(f"En iyi model yüklendi (epoch {best_epoch})")

        return self.history

    def _evaluate(self, loader: DataLoader,
                  criterion: nn.Module) -> tuple:
        """Validation/Test değerlendirmesi."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        tp, fp, fn = 0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)

                total_loss += loss.item() * X_batch.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

                tp += ((preds == 1) & (y_batch == 1)).sum().item()
                fp += ((preds == 1) & (y_batch == 0)).sum().item()
                fn += ((preds == 0) & (y_batch == 1)).sum().item()

        avg_loss = total_loss / total
        acc = correct / total
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)

        return avg_loss, acc, f1

    def predict(self, loader: DataLoader) -> tuple:
        """
        Tahmin üretir.

        Returns
        -------
        y_true, y_pred, y_prob : np.ndarray
        """
        self.model.eval()
        all_true = []
        all_pred = []
        all_prob = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                probs = torch.softmax(outputs, dim=1)

                all_true.append(y_batch.numpy())
                all_pred.append(outputs.argmax(dim=1).cpu().numpy())
                all_prob.append(probs[:, 1].cpu().numpy())

        return (np.concatenate(all_true),
                np.concatenate(all_pred),
                np.concatenate(all_prob))

    def _save_checkpoint(self, name: str, epoch: int,
                         val_loss: float, val_f1: float):
        path = self.checkpoint_dir / f"{name}_best.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_loss": val_loss,
            "val_f1": val_f1
        }, str(path))

    def _load_checkpoint(self, name: str):
        path = self.checkpoint_dir / f"{name}_best.pt"
        if path.exists():
            ckpt = torch.load(str(path), map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
