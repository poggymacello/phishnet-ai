import numpy as np
import pandas as pd

from phishnet import leakage


def test_single_feature_auc_is_half_for_uninformative_feature():
    labels = np.array([0, 1] * 50)
    features = pd.DataFrame({"noise": np.random.default_rng(0).random(100)})
    aucs = leakage.single_feature_auc(features, labels)
    assert 0.3 < aucs["noise"] < 0.7


def test_single_feature_auc_flags_a_circular_feature():
    labels = np.array([0] * 50 + [1] * 50)
    # perfectly encodes the label -- exactly the URLSimilarityIndex pattern
    features = pd.DataFrame({"circular": labels.astype(float)})
    aucs = leakage.single_feature_auc(features, labels)
    assert aucs["circular"] == 1.0


def test_flag_suspicious_features_respects_threshold():
    labels = np.array([0] * 50 + [1] * 50)
    features = pd.DataFrame(
        {
            "circular": labels.astype(float),
            "noise": np.random.default_rng(1).random(100),
        }
    )
    suspicious = leakage.flag_suspicious_features(features, labels, threshold=0.98)
    assert "circular" in suspicious.index
    assert "noise" not in suspicious.index


def test_constant_feature_scores_as_uninformative():
    labels = np.array([0, 1] * 20)
    features = pd.DataFrame({"const": np.zeros(40)})
    aucs = leakage.single_feature_auc(features, labels)
    assert aucs["const"] == 0.5


def test_duplicate_row_count_across_splits():
    train = pd.DataFrame({"a": [1, 2, 3]})
    val = pd.DataFrame({"a": [3, 4]})  # row "3" duplicated with train
    test = pd.DataFrame({"a": [5, 6]})
    assert leakage.duplicate_row_count_across_splits(train, val, test) == 1
