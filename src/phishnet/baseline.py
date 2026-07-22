"""TF-IDF + logistic regression baseline, used as a sanity comparison for the NN."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_baseline(seed: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )


def train_baseline(texts: list[str], labels: list[int], seed: int = 42) -> Pipeline:
    pipeline = build_baseline(seed=seed)
    pipeline.fit(texts, labels)
    return pipeline


def predict_proba(pipeline: Pipeline, texts: list[str]) -> np.ndarray:
    return pipeline.predict_proba(texts)[:, 1]
