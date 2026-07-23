"""Baselines the attention NN is judged against.

- TF-IDF over character n-grams (not words: URLs don't have "words" the
  way natural-language text does) + logistic regression.
- Gradient boosting over structured features, both a URL-only (deployable)
  variant and a full-feature (page-content included) upper-bound variant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_baseline(seed: int = 42) -> Pipeline:
    """Word-level TF-IDF + logistic regression, used by the v1 synthetic pipeline."""
    token_pattern = r"(?u)\b\w+\b"  # nosec B105 - a tokenizer regex, not a credential
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, token_pattern=token_pattern)),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )


def train_baseline(texts: list[str], labels: list[int], seed: int = 42) -> Pipeline:
    pipeline = build_baseline(seed=seed)
    pipeline.fit(texts, labels)
    return pipeline


def predict_proba(pipeline: Pipeline, texts: list[str]) -> np.ndarray:
    return pipeline.predict_proba(texts)[:, 1]


def build_char_tfidf_baseline(seed: int = 42) -> Pipeline:
    """Character n-gram TF-IDF + logistic regression, for real URL text."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")),
        ]
    )


def train_char_tfidf_baseline(urls: list[str], labels: list[int], seed: int = 42) -> Pipeline:
    pipeline = build_char_tfidf_baseline(seed=seed)
    pipeline.fit(urls, labels)
    return pipeline


def predict_proba_pipeline(pipeline: Pipeline, urls: list[str]) -> np.ndarray:
    return pipeline.predict_proba(urls)[:, 1]


def train_gradient_boosting(
    features: pd.DataFrame, labels: np.ndarray, seed: int = 42
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(random_state=seed, class_weight="balanced")
    model.fit(features, labels)
    return model


def predict_proba_sklearn(model, features: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(features)[:, 1]
