"""Evaluation metrics, comparison plots, and an attention-meaningfulness check."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Tokens that show up disproportionately in the phishing templates (urgency,
# money, credential/verification language). Used only to sanity-check what
# the attention mechanism is actually attending to, not as model features.
TRIGGER_WORDS = {
    "klik",
    "verifikasi",
    "segera",
    "menang",
    "bayar",
    "blokir",
    "terblokir",
    "bonus",
    "gagal",
    "hubungi",
    "kadaluarsa",
    "konfirmasi",
}


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: np.ndarray = field(repr=False)

    def as_dict(self) -> dict[str, float]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
        }


def compute_metrics(y_true: list[int], y_prob: np.ndarray, threshold: float = 0.5) -> Metrics:
    y_pred = (y_prob >= threshold).astype(int)
    return Metrics(
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_prob),
        confusion=confusion_matrix(y_true, y_pred),
    )


def plot_roc_curves(results: dict[str, tuple[list[int], np.ndarray]], out_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    for name, (y_true, y_prob) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("ROC: attention model vs TF-IDF baseline")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        plt.text(j, i, str(value), ha="center", va="center")
    plt.xticks([0, 1], ["normal", "phishing"])
    plt.yticks([0, 1], ["normal", "phishing"])
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_attention_heatmap(attn_matrix: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(attn_matrix, cmap="hot")
    plt.colorbar()
    plt.title("attention weights for a sample phishing email")
    plt.xlabel("token position")
    plt.ylabel("token position")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def attention_trigger_score(
    attn_matrices: np.ndarray, token_lists: list[list[str]]
) -> dict[str, float]:
    """Compare average attention received by trigger-word tokens vs other tokens.

    For each sample, attention received by position ``j`` is the column mean
    of the attention matrix (how much the rest of the sequence attends to
    token j). We then split those per-token scores into trigger-word tokens
    and everything else, and report the mean of each group. This is a
    plain empirical check, not a causal claim.
    """
    trigger_scores: list[float] = []
    other_scores: list[float] = []
    for attn, tokens in zip(attn_matrices, token_lists, strict=False):
        received = attn.mean(axis=0)  # column means: attention received per position
        for pos, token in enumerate(tokens):
            score = float(received[pos])
            if token in TRIGGER_WORDS:
                trigger_scores.append(score)
            else:
                other_scores.append(score)

    trigger_mean = float(np.mean(trigger_scores)) if trigger_scores else float("nan")
    other_mean = float(np.mean(other_scores)) if other_scores else float("nan")
    return {
        "trigger_mean_attention": round(trigger_mean, 4),
        "other_mean_attention": round(other_mean, 4),
        "n_trigger_tokens": len(trigger_scores),
        "n_other_tokens": len(other_scores),
    }
