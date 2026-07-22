import numpy as np

from phishnet.evaluate import attention_trigger_score, compute_metrics


def test_compute_metrics_matches_hand_worked_confusion_matrix():
    # 2 true positives, 1 false negative, 1 false positive, 4 true negatives
    y_true = [1, 1, 1, 0, 0, 0, 0, 0]
    y_prob = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1])

    metrics = compute_metrics(y_true, y_prob, threshold=0.5)

    # predicted positives at threshold 0.5: indices 0,1,3 -> TP=2, FP=1, FN=1
    expected_precision = 2 / 3
    expected_recall = 2 / 3
    expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)

    assert abs(metrics.precision - expected_precision) < 1e-6
    assert abs(metrics.recall - expected_recall) < 1e-6
    assert abs(metrics.f1 - expected_f1) < 1e-6
    assert metrics.confusion.sum() == len(y_true)


def test_attention_trigger_score_separates_groups():
    # two fake "samples": attention matrix where the trigger token clearly
    # receives more attention (higher column values) than the other token
    attn = np.array(
        [
            [[0.1, 0.9], [0.1, 0.9]],  # column means: token0=0.1, token1=0.9
        ]
    )
    tokens = [["halo", "klik"]]  # "klik" is a trigger word, "halo" is not

    stats = attention_trigger_score(attn, tokens)
    assert stats["n_trigger_tokens"] == 1
    assert stats["n_other_tokens"] == 1
    assert stats["trigger_mean_attention"] > stats["other_mean_attention"]
