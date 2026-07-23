from pathlib import Path

from phishnet.real_pipeline import duplicate_check, run_full_pipeline

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "phiusiil_sample.csv"


def test_run_full_pipeline_on_sample_produces_all_models():
    result = run_full_pipeline(seed=1, sample_csv=str(SAMPLE_PATH))
    expected_models = {
        "Attention NN (char, URL-only)",
        "TF-IDF char n-gram + LogReg (URL-only)",
        "Gradient Boosting (URL-only)",
        "Gradient Boosting (full features, upper bound)",
    }
    assert set(result["scores"].keys()) == expected_models
    assert set(result["metrics"].keys()) == expected_models

    for m in result["metrics"].values():
        assert 0.0 <= m["roc_auc"] <= 1.0
        assert "recall_at_fpr_1pct" in m
        assert "recall_at_fpr_5pct" in m
        assert "bootstrap_pr_auc_ci" in m


def test_run_full_pipeline_reports_base_rates_per_split():
    result = run_full_pipeline(seed=1, sample_csv=str(SAMPLE_PATH))
    for split_name in ("train", "val", "test"):
        rate = result["base_rates"][split_name]
        assert 0.0 < rate < 1.0


def test_no_domain_overlap_across_splits():
    result = run_full_pipeline(seed=1, sample_csv=str(SAMPLE_PATH))
    dup_count = duplicate_check(result["train_full"], result["val"], result["test"])
    assert dup_count == 0


def test_leakage_audit_flags_no_url_only_feature():
    # After dropping URLSimilarityIndex (see README's Leakage Controls),
    # only page-content features (LineOfCode, NoOfExternalRef, ...) may
    # still cross the suspicious-AUC threshold, and those were manually
    # investigated and kept as genuine signal, not leakage -- no URL-only
    # (pre-fetch, actually deployed) feature should ever be in this set.
    from phishnet.data_real import URL_ONLY_FEATURES

    result = run_full_pipeline(seed=1, sample_csv=str(SAMPLE_PATH))
    suspicious = result["leakage_audit"]["suspicious_features"]
    assert not set(suspicious).intersection(URL_ONLY_FEATURES)
