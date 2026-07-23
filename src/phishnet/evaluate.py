"""Evaluation metrics, comparison plots, and an attention-meaningfulness check."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Characters that show up disproportionately in obfuscated/suspicious URLs:
# digits (lookalike domains, IP-literal hosts), and common delimiter/
# obfuscation characters. Used only to sanity-check what the attention
# mechanism attends to on real URL text, not as model features.
SUSPICIOUS_URL_CHARS = set("0123456789@%-_=&?")

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


def recall_at_fpr(y_true, scores: np.ndarray, max_fpr: float) -> float:
    """Highest recall (TPR) achievable while keeping FPR at or below ``max_fpr``.

    This is the operating-point metric that actually matters for a
    deployed detector: a SOC or a browser warning has a fixed tolerance
    for false positives, and AUC alone doesn't say what recall you get
    once you respect that budget.
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = fpr <= max_fpr
    if not valid.any():
        return 0.0
    return float(tpr[valid].max())


def polarity_warning(roc_auc: float) -> str | None:
    """Flags a likely inverted score/label, not just a weak model.

    A genuinely weak but correctly-oriented model still centers near 0.5;
    a ROC-AUC meaningfully *below* 0.5 (not just noisy-around-0.5) is a
    much stronger signal of an inverted label or score than of poor
    learning, and is worth an automatic flag rather than a silent low
    number in a metrics table.
    """
    if roc_auc < 0.4:
        return (
            f"ROC-AUC={roc_auc:.4f} is well below 0.5 -- check for an inverted "
            f"label or score polarity before concluding the model is just weak "
            f"(1 - {roc_auc:.4f} = {1 - roc_auc:.4f})"
        )
    return None


def bootstrap_ci(
    y_true, scores: np.ndarray, n_bootstrap: int = 1000, seed: int = 42, confidence: float = 0.95
) -> dict[str, float]:
    """Bootstrap confidence interval for PR-AUC."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)

    point_estimate = float(average_precision_score(y_true, scores))
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_sample = y_true[idx]
        if y_sample.min() == y_sample.max():
            continue
        samples.append(average_precision_score(y_sample, scores[idx]))

    alpha = (1 - confidence) / 2
    return {
        "point_estimate": round(point_estimate, 4),
        "ci_lower": round(float(np.quantile(samples, alpha)), 4),
        "ci_upper": round(float(np.quantile(samples, 1 - alpha)), 4),
    }


def url_attention_trigger_score(
    attn_matrices: np.ndarray, token_lists: list[list[str]]
) -> dict[str, float]:
    """Real-URL analog of ``attention_trigger_score``: compares attention
    received by suspicious URL characters (digits, obfuscation characters)
    against everything else, instead of Indonesian phishing trigger words.
    """
    trigger_scores: list[float] = []
    other_scores: list[float] = []
    for attn, tokens in zip(attn_matrices, token_lists, strict=False):
        received = attn.mean(axis=0)
        for pos, token in enumerate(tokens):
            if pos >= len(received):
                break
            score = float(received[pos])
            if token in SUSPICIOUS_URL_CHARS:
                trigger_scores.append(score)
            else:
                other_scores.append(score)

    trigger_mean = float(np.mean(trigger_scores)) if trigger_scores else float("nan")
    other_mean = float(np.mean(other_scores)) if other_scores else float("nan")
    return {
        "trigger_mean_attention": round(trigger_mean, 4),
        "other_mean_attention": round(other_mean, 4),
        "n_trigger_chars": len(trigger_scores),
        "n_other_chars": len(other_scores),
    }


def plot_roc_curves(results: dict[str, tuple[list[int], np.ndarray]], out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    for name, (y_true, y_prob) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("ROC: " + " vs ".join(results.keys()))
    plt.legend(loc="lower right", fontsize=8)
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
