"""End-to-end pipeline for the real PhiUSIIL data: load, split, audit, fit, evaluate.

Three "deployable" (URL-only, pre-fetch) models are compared on an equal
training-set size: the attention NN, TF-IDF character n-gram + logistic
regression, and gradient boosting on structured URL-only features. A
fourth model -- gradient boosting on the full feature set, including
page-content features that require fetching the page -- is trained
separately on the full training set, as an upper-bound reference rather
than a fair fourth competitor (see README's Method section for why the
comparison is scoped this way).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from phishnet import baseline as baseline_mod
from phishnet import evaluate as eval_mod
from phishnet import leakage
from phishnet import model as model_mod
from phishnet.data_real import (
    CharVocabulary,
    UrlDataset,
    group_split,
    load_raw,
    prepare,
)

MAX_ACCEPTABLE_FPR = (0.01, 0.05)
# see module docstring: full-batch attention on 160k rows is impractical
NN_TRAIN_SUBSAMPLE = 20000
CHAR_MAX_LEN = 60


def _stratified_subsample(dataset: UrlDataset, n: int, seed: int) -> UrlDataset:
    if len(dataset) <= n:
        return dataset
    rng = np.random.default_rng(seed)
    idx_pos = np.where(dataset.labels == 1)[0]
    idx_neg = np.where(dataset.labels == 0)[0]
    frac = n / len(dataset)
    n_pos = int(round(len(idx_pos) * frac))
    n_neg = n - n_pos
    chosen = np.concatenate(
        [
            rng.choice(idx_pos, size=min(n_pos, len(idx_pos)), replace=False),
            rng.choice(idx_neg, size=min(n_neg, len(idx_neg)), replace=False),
        ]
    )
    rng.shuffle(chosen)
    return UrlDataset(
        url_only_features=dataset.url_only_features.iloc[chosen].reset_index(drop=True),
        page_content_features=dataset.page_content_features.iloc[chosen].reset_index(drop=True),
        labels=dataset.labels[chosen],
        urls=dataset.urls.iloc[chosen].reset_index(drop=True),
        etld1=dataset.etld1.iloc[chosen].reset_index(drop=True),
    )


def run_full_pipeline(seed: int = 42, sample_csv: str | None = None) -> dict:
    df = pd.read_csv(sample_csv) if sample_csv is not None else load_raw()
    dataset = prepare(df)
    train_full, val, test = group_split(dataset, seed=seed)

    audit = _run_leakage_audit(train_full)
    train = _stratified_subsample(train_full, NN_TRAIN_SUBSAMPLE, seed=seed)

    # --- attention NN (character-level URL text) ---
    vocab = CharVocabulary(max_len=CHAR_MAX_LEN).fit(train.urls)
    X_train_nn = vocab.encode(train.urls)
    X_test_nn = vocab.encode(test.urls)

    nn_start = time.perf_counter()
    nn_model = model_mod.AttentionClassifier(vocab_size=vocab.size)
    loss_history = model_mod.train_model(nn_model, X_train_nn, list(train.labels), seed=seed)
    nn_fit_seconds = time.perf_counter() - nn_start
    nn_scores = model_mod.predict_proba(nn_model, X_test_nn)

    # --- TF-IDF character n-gram + logistic regression (deployable) ---
    char_tfidf_pipeline = baseline_mod.train_char_tfidf_baseline(
        list(train.urls), list(train.labels), seed=seed
    )
    tfidf_scores = baseline_mod.predict_proba_pipeline(char_tfidf_pipeline, list(test.urls))

    # --- gradient boosting on URL-only structured features (deployable) ---
    gbm_url_only = baseline_mod.train_gradient_boosting(
        train.url_only_features, train.labels, seed=seed
    )
    gbm_url_only_scores = baseline_mod.predict_proba_sklearn(gbm_url_only, test.url_only_features)

    # --- gradient boosting on the FULL feature set incl. page content (upper bound, full train) ---
    full_train_features = pd.concat(
        [train_full.url_only_features, train_full.page_content_features], axis=1
    )
    full_test_features = pd.concat([test.url_only_features, test.page_content_features], axis=1)
    gbm_full = baseline_mod.train_gradient_boosting(
        full_train_features, train_full.labels, seed=seed
    )
    gbm_full_scores = baseline_mod.predict_proba_sklearn(gbm_full, full_test_features)

    results = {
        "Attention NN (char, URL-only)": nn_scores,
        "TF-IDF char n-gram + LogReg (URL-only)": tfidf_scores,
        "Gradient Boosting (URL-only)": gbm_url_only_scores,
        "Gradient Boosting (full features, upper bound)": gbm_full_scores,
    }

    metrics = {}
    polarity_flags = {}
    for name, scores in results.items():
        m = eval_mod.compute_metrics(test.labels, scores)
        metrics[name] = m.as_dict()
        for fpr_budget in MAX_ACCEPTABLE_FPR:
            recall = eval_mod.recall_at_fpr(test.labels, scores, fpr_budget)
            metrics[name][f"recall_at_fpr_{int(fpr_budget * 100)}pct"] = round(recall, 4)
        metrics[name]["bootstrap_pr_auc_ci"] = eval_mod.bootstrap_ci(test.labels, scores, seed=seed)
        polarity_flags[name] = eval_mod.polarity_warning(m.roc_auc)

    attn = model_mod.attention_weights(nn_model, X_test_nn[:30])
    token_lists = [vocab.tokens(u) for u in test.urls[:30]]
    trigger_stats = eval_mod.url_attention_trigger_score(attn, token_lists)

    return {
        "train_full": train_full,
        "train_subsample": train,
        "val": val,
        "test": test,
        "leakage_audit": audit,
        "nn_model": nn_model,
        "nn_fit_seconds": nn_fit_seconds,
        "nn_loss_history": loss_history,
        "vocab": vocab,
        "scores": results,
        "metrics": metrics,
        "polarity_flags": polarity_flags,
        "trigger_stats": trigger_stats,
        "base_rates": {
            "train": float(train_full.labels.mean()),
            "val": float(val.labels.mean()),
            "test": float(test.labels.mean()),
        },
    }


def _run_leakage_audit(train: UrlDataset) -> dict:
    all_features = pd.concat([train.url_only_features, train.page_content_features], axis=1)
    aucs = leakage.single_feature_auc(all_features, train.labels)
    suspicious = leakage.flag_suspicious_features(all_features, train.labels)
    return {
        "top_single_feature_aucs": aucs.head(8).round(4).to_dict(),
        "suspicious_features": suspicious.round(4).to_dict(),
    }


def duplicate_check(train: UrlDataset, val: UrlDataset, test: UrlDataset) -> int:
    all_train = pd.concat([train.url_only_features, train.page_content_features], axis=1)
    all_val = pd.concat([val.url_only_features, val.page_content_features], axis=1)
    all_test = pd.concat([test.url_only_features, test.page_content_features], axis=1)
    return leakage.duplicate_row_count_across_splits(all_train, all_val, all_test)
