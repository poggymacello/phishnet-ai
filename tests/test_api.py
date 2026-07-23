from pathlib import Path

import pytest
import tomllib
from fastapi.testclient import TestClient

from phishnet import api
from phishnet.artifact import build_artifact

URLS = [
    "https://accounts.google.com/signin",
    "http://secure-paypal-verify-account.tk/login.php",
    "https://en.wikipedia.org/wiki/Phishing",
    "http://192.168.1.1.confirm-billing.info/update",
] * 10
LABELS = [0, 1, 0, 1] * 10


@pytest.fixture
def client():
    api._artifact = build_artifact(URLS, LABELS, seed=42)
    yield TestClient(api.app)
    api._artifact = None


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_model_info(client):
    resp = client.get("/model")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == api.__version__
    assert "library_versions" in body


def test_predict_valid_url(client):
    resp = client.post("/predict", json={"url": "https://example.com/page"})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["predicted_label"] in (0, 1)
    assert body["model_version"] == api.__version__


def test_predict_rejects_empty_url(client):
    resp = client.post("/predict", json={"url": ""})
    assert resp.status_code == 422


def test_predict_rejects_oversized_url(client):
    resp = client.post("/predict", json={"url": "http://" + "a" * 3000})
    assert resp.status_code == 422


def test_predict_rejects_missing_field(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_metrics_endpoint_exposes_prometheus_format(client):
    client.get("/healthz")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "phishnet_requests_total" in resp.text


def test_model_version_matches_pyproject(client):
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        declared_version = tomllib.load(f)["project"]["version"]

    resp = client.get("/model")
    assert resp.json()["version"] == declared_version
