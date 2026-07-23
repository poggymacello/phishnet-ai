"""Automated leakage checks: single-feature predictive power and cross-split duplicates.

Run against every candidate feature before trusting any model result. A
single feature with near-perfect AUC against the label is a strong signal
of leakage (the feature is encoding the label, directly or by proxy)
rather than a genuinely predictive behavioral signal -- but the audit only
tells you where to look, not what to conclude. See README's Leakage
Controls section for the manual investigation that followed each flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SUSPICIOUS_AUC_THRESHOLD = 0.98


def single_feature_auc(features: pd.DataFrame, labels: np.ndarray) -> pd.Series:
    """AUC of each feature alone (as a raw score) against the label.

    A feature entirely unrelated to the label gives ~0.5. Values far from
    0.5 in either direction (a feature can be informative "backwards") are
    both meaningful, so this reports max(auc, 1 - auc).
    """
    scores = {}
    for col in features.columns:
        values = features[col].to_numpy()
        if np.all(values == values[0]):
            scores[col] = 0.5
            continue
        auc = roc_auc_score(labels, values)
        scores[col] = max(auc, 1 - auc)
    return pd.Series(scores).sort_values(ascending=False)


def flag_suspicious_features(
    features: pd.DataFrame, labels: np.ndarray, threshold: float = SUSPICIOUS_AUC_THRESHOLD
) -> pd.Series:
    aucs = single_feature_auc(features, labels)
    return aucs[aucs > threshold]


def duplicate_row_count_across_splits(*feature_frames: pd.DataFrame) -> int:
    """Count rows (by exact feature-value hash) that appear in more than one split."""
    seen: dict[int, set[int]] = {}
    for split_idx, frame in enumerate(feature_frames):
        row_hashes = pd.util.hash_pandas_object(frame, index=False)
        for h in row_hashes:
            seen.setdefault(int(h), set()).add(split_idx)
    return sum(1 for splits in seen.values() if len(splits) > 1)
