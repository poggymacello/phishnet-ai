# Data

## Real data: PhiUSIIL Phishing URL Dataset

The primary dataset is [PhiUSIIL](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
(Prasad & Chandra, 2024), 235,795 URLs (134,850 legitimate, 100,945
phishing) with 54 precomputed features covering both URL-string
properties and rendered-page content.

- **Source**: `https://archive.ics.uci.edu/static/public/967/data.csv`,
  UCI's own static CDN. Verified live via UCI's dataset API
  (`https://archive.ics.uci.edu/api/dataset?id=967`) before writing
  `scripts/download_data.py` -- unlike shadowtrace's UNSW-NB15, no
  third-party mirror was needed here.
- **License**: CC BY 4.0 (per the UCI dataset page).
- **Citation**: Prasad, A. and Chandra, S. "PhiUSIIL: A diverse security
  profile empowered phishing URL detection framework based on similarity
  index and incremental learning." Computers & Security, 2024.
- **SHA256** (pinned in `scripts/download_data.py`):
  `a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6`
- **Access date**: 2026-07-23.
- Not committed to git (~57MB). Fetch it with:

  ```bash
  python scripts/download_data.py
  ```

### Label convention

PhiUSIIL uses `label=1` for legitimate and `label=0` for phishing. This
project (and every other repo in this portfolio) uses the opposite
convention, `1` = the thing being detected, so `phishnet.data_real.prepare`
flips it: `labels = 1 - df["label"]`.

### Feature split: URL-only vs. page-content

`phishnet.data_real` splits the 49 usable features into two groups, not
one, because they have different operational realism:

- `URL_ONLY_FEATURES` (21 columns): derivable from the URL string alone,
  before ever connecting to the site -- what a real pre-click filter
  could actually use, and what the deployed model at `/predict` computes.
- `PAGE_CONTENT_FEATURES` (28 columns): derived from the rendered page
  (line count, image count, title match score, ...). Real signal, but
  only available after fetching the page. Reported as an upper-bound
  comparison in the README's Results section, never deployed -- see
  Deployment for why a live URL-fetching endpoint is a straightforward
  SSRF vector this project avoids by design.

`FILENAME`, `URL`, `Domain`, `TLD`, `Title`, and `URLSimilarityIndex` are
excluded from both feature sets: the first four are identifiers, not
features, and `URLSimilarityIndex` was dropped as leakage (see README's
Leakage Controls section -- it is exactly 100.0 for every legitimate row).

## Sample fixture

`data/sample/phiusiil_sample.csv` is a small (3,000-row, ~620KB),
class-stratified (1,500/1,500) sample of the full dataset, generated with
a fixed seed (42) and committed to git. It exists purely so tests and CI
can exercise the full real-data pipeline (loading, group-splitting,
leakage audit, model fitting, metrics) without downloading the full 57MB
file. It is not used for any reported result in the README -- those come
from the full dataset via `python -m phishnet real-train`.

## Synthetic fallback (v1)

The original synthetic generator (`src/phishnet/data.py`, 12 phishing +
12 normal templates) is still present and still exercised by the `train`/
`eval` CLI commands and their tests, kept as a fast, dependency-free
smoke test of the attention model and baseline independent of the real
dataset. See the README's "What changed from v1" section for why it's
kept rather than deleted.
