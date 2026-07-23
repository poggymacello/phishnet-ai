"""Download the PhiUSIIL phishing URL dataset from UCI and verify its integrity.

UCI's dataset API (https://archive.ics.uci.edu/api/dataset?id=967) was
queried before writing this script and returned a live, stable ``data_url``
pointing at UCI's own static CDN -- no third-party mirror was needed here,
unlike shadowtrace's UNSW-NB15 (whose original host is dead). If that ever
changes, this script fails loudly on a checksum mismatch rather than
silently serving different data than what this project's README and
figures describe.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

URL = "https://archive.ics.uci.edu/static/public/967/data.csv"
DEST_NAME = "phiusiil.csv"
SHA256 = "a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6"


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / DEST_NAME

    if dest.exists() and _sha256_of(dest) == SHA256:
        print(f"{DEST_NAME}: already present and verified, skipping")
        return

    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, dest)  # noqa: S310 (fixed https URL, checksum-verified below)

    actual_sha256 = _sha256_of(dest)
    if actual_sha256 != SHA256:
        dest.unlink()
        print(
            f"FATAL: {DEST_NAME} checksum mismatch "
            f"(expected {SHA256}, got {actual_sha256}). "
            "UCI's hosted copy may have changed. Not proceeding with "
            "unverified data -- see data/README.md before changing the "
            "pinned checksum.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"{DEST_NAME}: downloaded and verified ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
