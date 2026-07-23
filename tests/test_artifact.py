from pathlib import Path

from phishnet.artifact import (
    artifact_path_for_version,
    build_artifact,
    load_artifact,
    save_artifact,
)

URLS = [
    "https://accounts.google.com/signin",
    "http://secure-paypal-verify-account.tk/login.php",
    "https://en.wikipedia.org/wiki/Phishing",
    "http://192.168.1.1.confirm-billing.info/update",
] * 10
LABELS = [0, 1, 0, 1] * 10


def test_build_artifact_scores_are_probabilities():
    artifact = build_artifact(URLS, LABELS, seed=42)
    scores = artifact.score(URLS[:4])
    assert len(scores) == 4
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_artifact_round_trips_through_disk(tmp_path):
    artifact = build_artifact(URLS, LABELS, seed=42)
    path = tmp_path / "test.joblib"
    save_artifact(artifact, path)
    assert path.exists()

    loaded = load_artifact(path)
    assert loaded.version == artifact.version
    assert list(loaded.score(URLS[:2])) == list(artifact.score(URLS[:2]))


def test_artifact_path_for_version():
    path = artifact_path_for_version(Path("models"), version="9.9.9")
    assert path == Path("models") / "phishnet-9.9.9.joblib"


def test_top_contributing_ngrams_returns_ranked_list():
    artifact = build_artifact(URLS, LABELS, seed=42)
    contributions = artifact.top_contributing_ngrams(URLS[0], top_k=3)
    assert len(contributions) <= 3
    magnitudes = [abs(c["contribution"]) for c in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_top_contributing_ngrams_empty_for_unseen_characters():
    artifact = build_artifact(URLS, LABELS, seed=42)
    contributions = artifact.top_contributing_ngrams("中文域名", top_k=3)
    assert contributions == []
