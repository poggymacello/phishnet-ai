"""Versioned model artifact for the deployed real-data model.

Only the character n-gram TF-IDF + logistic regression pipeline is bundled
here, not the structured-feature gradient boosting model. The GBM's 21
URL-only features (TLDLegitimateProb, CharContinuationRate, URLCharProb,
...) come from PhiUSIIL's own precomputed feature-extraction pipeline,
which isn't reproduced in this project -- there's no code here that turns
an arbitrary raw URL into those exact feature values. The TF-IDF model
needs nothing but the URL string itself, so it's the one that can honestly
be deployed as a "give me a URL, get a score" service. See README's
Deployment section.
"""

from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline

from phishnet import __version__
from phishnet import baseline as baseline_mod

RAW_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "phiusiil.csv"


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_data_hash() -> str:
    if not RAW_PATH.exists():
        return "unavailable (trained without local raw data present)"
    return _sha256_of(RAW_PATH)


@dataclass
class ModelArtifact:
    version: str
    trained_at: str
    seed: int
    pipeline: Pipeline
    training_data_sha256: str
    library_versions: dict[str, str] = field(default_factory=dict)
    python_version: str = field(default_factory=platform.python_version)

    def score(self, urls: list[str]) -> np.ndarray:
        return baseline_mod.predict_proba_pipeline(self.pipeline, urls)

    def top_contributing_ngrams(self, url: str, top_k: int = 5) -> list[dict]:
        """N-grams from this URL with the largest signed contribution to the
        score (tfidf weight * logistic-regression coefficient), positive
        meaning "pushes toward phishing"."""
        tfidf = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]
        vec = tfidf.transform([url]).toarray()[0]
        contributions = vec * clf.coef_[0]
        nonzero = np.nonzero(vec)[0]
        if len(nonzero) == 0:
            return []
        feature_names = tfidf.get_feature_names_out()
        ranked = sorted(nonzero, key=lambda i: abs(contributions[i]), reverse=True)[:top_k]
        return [
            {"ngram": feature_names[i], "contribution": round(float(contributions[i]), 4)}
            for i in ranked
        ]


def build_artifact(urls: list[str], labels: list[int], seed: int = 42) -> ModelArtifact:
    pipeline = baseline_mod.train_char_tfidf_baseline(urls, labels, seed=seed)
    return ModelArtifact(
        version=__version__,
        trained_at=datetime.now(timezone.utc).isoformat(),
        seed=seed,
        pipeline=pipeline,
        training_data_sha256=_training_data_hash(),
        library_versions={
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
        },
    )


def save_artifact(artifact: ModelArtifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: Path) -> ModelArtifact:
    return joblib.load(path)


def artifact_path_for_version(models_dir: Path, version: str | None = None) -> Path:
    version = version or __version__
    return models_dir / f"phishnet-{version}.joblib"
