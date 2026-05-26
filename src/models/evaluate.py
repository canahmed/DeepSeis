"""
DeepSeis - Model Değerlendirme Modülü
=======================================

Eğitilmiş modelleri standart metriklerle değerlendirir ve
tez kalitesinde görseller üretir.

Metrikler: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
Görseller: Confusion Matrix, ROC Eğrisi, Eğitim Kayıp Eğrileri

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import logging
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

logger = logging.getLogger("DeepSeis.Evaluate")


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray) -> dict:
    """
    Tüm değerlendirme metriklerini hesaplar.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "support": {
            "total": len(y_true),
            "positive": int(y_true.sum()),
            "negative": int(len(y_true) - y_true.sum())
        }
    }

    return metrics


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          model_name: str = "Model",
                          save_path: str = None) -> plt.Figure:
    """Confusion matrix görselleştirmesi."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    # Değerleri hücrelere yaz
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i,j]:,}",
                    ha="center", va="center", color=color, fontsize=16)

    ax.set_xlabel("Tahmin", fontsize=12)
    ax.set_ylabel("Gerçek", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal (0)", "Anomali (1)"])
    ax.set_yticklabels(["Normal (0)", "Anomali (1)"])
    ax.set_title(f"Karışıklık Matrisi — {model_name}", fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close()
    return fig


def plot_roc_curve(y_true: np.ndarray,
                   y_prob: np.ndarray,
                   model_name: str = "Model",
                   save_path: str = None) -> plt.Figure:
    """ROC eğrisi."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"{model_name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Rastgele")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2563eb")

    ax.set_xlabel("Yanlış Pozitif Oranı (FPR)")
    ax.set_ylabel("Doğru Pozitif Oranı (TPR)")
    ax.set_title(f"ROC Eğrisi — {model_name}")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close()
    return fig


def plot_training_history(history: dict,
                          model_name: str = "Model",
                          save_path: str = None) -> plt.Figure:
    """Eğitim kayıp ve metrik eğrileri."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train", lw=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val", lw=2)
    axes[0].set_title("Kayıp (Loss)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train", lw=2)
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val", lw=2)
    axes[1].set_title("Doğruluk (Accuracy)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1
    axes[2].plot(epochs, history["train_f1"], "b-", label="Train", lw=2)
    axes[2].plot(epochs, history["val_f1"], "r-", label="Val", lw=2)
    axes[2].set_title("F1-Score")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Eğitim Geçmişi — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close()
    return fig


def plot_model_comparison(results: dict,
                          save_path: str = None) -> plt.Figure:
    """Tüm modellerin metriklerini karşılaştırır."""
    models = list(results.keys())
    metrics_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    labels_tr = ["Doğruluk", "Kesinlik", "Duyarlılık", "F1-Skor", "ROC-AUC"]

    x = np.arange(len(metrics_names))
    width = 0.8 / len(models)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#8b5cf6"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model_name in enumerate(models):
        vals = [results[model_name].get(m, 0) for m in metrics_names]
        ax.bar(x + i * width, vals, width, label=model_name,
               color=colors[i % len(colors)], alpha=0.85)

        # Değer etiketleri
        for j, v in enumerate(vals):
            ax.text(x[j] + i * width, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(labels_tr)
    ax.set_ylabel("Skor")
    ax.set_title("Model Karşılaştırması", fontsize=14)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close()
    return fig


def full_evaluation(y_true, y_pred, y_prob,
                    model_name: str, figures_dir: str,
                    history: dict = None,
                    threshold: float = None,
                    threshold_source: str = None,
                    validation_f1_at_threshold: float = None,
                    experiment_name: str = None) -> dict:
    """
    Tam değerlendirme: metrikler + tüm görseller.
    """
    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)

    # Metrikler
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["threshold"] = threshold
    metrics["threshold_source"] = threshold_source
    metrics["validation_f1_at_threshold"] = validation_f1_at_threshold
    metrics["experiment"] = experiment_name

    logger.info(f"\n{'='*50}")
    logger.info(f"DEĞERLENDİRME: {model_name}")
    logger.info(f"  Accuracy  : {metrics['accuracy']:.4f}")
    logger.info(f"  Precision : {metrics['precision']:.4f}")
    logger.info(f"  Recall    : {metrics['recall']:.4f}")
    logger.info(f"  F1-Score  : {metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC    : {metrics['pr_auc']:.4f}")
    logger.info(f"{'='*50}")

    # Görseller
    plot_confusion_matrix(y_true, y_pred, model_name,
                          str(figures / f"{model_name}_confusion_matrix.png"))
    plot_roc_curve(y_true, y_prob, model_name,
                   str(figures / f"{model_name}_roc_curve.png"))

    if history:
        plot_training_history(history, model_name,
                              str(figures / f"{model_name}_training_history.png"))

    # Rapor kaydet
    report_path = figures.parent / "reports" / f"{model_name}_metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
