"""FastAPI inference service for the deployed PhishNet model artifact.

Security notes (see SECURITY.md for the full policy):
- The submitted URL is never fetched. Scoring uses only the character
  n-gram statistics of the URL string itself, so this endpoint cannot be
  used to make the server issue requests to attacker-chosen hosts (SSRF).
- Input is validated strictly with Pydantic (type, length cap) before it
  reaches the model.
- The raw URL is never logged, only request metadata (status, latency),
  since a submitted URL can itself carry sensitive information (internal
  hostnames, tokens in query strings).
- Rate limiting caps how fast a single client can call the API.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from phishnet import __version__
from phishnet.artifact import ModelArtifact, artifact_path_for_version, load_artifact

logger = logging.getLogger("phishnet.api")
logging.basicConfig(level=logging.INFO)

MAX_URL_LENGTH = 2048

REQUEST_COUNT = Counter("phishnet_requests_total", "Total requests", ["endpoint", "status"])
PREDICT_LATENCY = Histogram("phishnet_predict_latency_seconds", "Predict latency")

_RATE_LIMIT = os.environ.get("PHISHNET_RATE_LIMIT", "120/minute")
limiter = Limiter(key_func=get_remote_address, default_limits=[_RATE_LIMIT])

app = FastAPI(title="PhishNet AI", version=__version__)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_artifact: ModelArtifact | None = None


def get_artifact() -> ModelArtifact:
    global _artifact
    if _artifact is None:
        model_path = Path(
            os.environ.get("PHISHNET_MODEL_PATH", str(artifact_path_for_version(Path("models"))))
        )
        _artifact = load_artifact(model_path)
        logger.info(
            "loaded model artifact version=%s trained_at=%s",
            _artifact.version,
            _artifact.trained_at,
        )
    return _artifact


class PredictRequest(BaseModel):
    url: str = Field(min_length=1, max_length=MAX_URL_LENGTH)


class ContributingFeature(BaseModel):
    ngram: str
    contribution: float


class PredictResponse(BaseModel):
    score: float
    predicted_label: int
    contributing_features: list[ContributingFeature]
    model_version: str


@app.middleware("http")
async def _timing_and_metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    REQUEST_COUNT.labels(endpoint=request.url.path, status=response.status_code).inc()
    if request.url.path == "/predict":
        PREDICT_LATENCY.observe(elapsed)
    logger.info(  # metadata only: no request body/URL is ever logged
        "request path=%s status=%s latency_ms=%.2f",
        request.url.path,
        response.status_code,
        elapsed * 1000,
    )
    return response


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model")
def model_info() -> dict:
    artifact = get_artifact()
    return {
        "version": artifact.version,
        "trained_at": artifact.trained_at,
        "seed": artifact.seed,
        "training_data_sha256": artifact.training_data_sha256,
        "library_versions": artifact.library_versions,
    }


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
@limiter.limit(_RATE_LIMIT)
def predict(request: Request, body: PredictRequest) -> PredictResponse:
    artifact = get_artifact()
    threshold = 0.5

    with PREDICT_LATENCY.time():
        score = float(artifact.score([body.url])[0])

    contributing = artifact.top_contributing_ngrams(body.url)

    logger.info("predict served score=%.4f", score)  # no URL in the log line
    return PredictResponse(
        score=score,
        predicted_label=int(score >= threshold),
        contributing_features=[ContributingFeature(**c) for c in contributing],
        model_version=artifact.version,
    )
