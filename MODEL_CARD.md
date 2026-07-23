# Model Card: PhishNet AI

## Purpose

Detects phishing URLs. Two generations exist in this repository:

- **v2 (deployed, real data)**: character n-gram TF-IDF + logistic
  regression, gradient boosting on structured URL features, and a small
  attention-based neural network, all trained and evaluated on the real
  PhiUSIIL dataset with a domain-grouped (eTLD+1) split. The TF-IDF model
  is the one actually served at `/predict`.
- **v1 (synthetic, kept for the fast pipeline demo)**: the same attention
  model against a TF-IDF baseline on a synthetic email-text generator.
  See README's "What changed from v1."

Portfolio demonstration of a real-data modeling and deployment pipeline
(leakage auditing, domain-grouped splitting, operating-point reporting,
containerized serving), not a production phishing filter.

## Data

PhiUSIIL Phishing URL Dataset, 235,795 real URLs (UCI, CC BY 4.0). See
[`data/README.md`](data/README.md) for source, license, citation, and the
URL-only vs. page-content feature split. Split by eTLD+1 (no domain's
URLs cross train/val/test) rather than randomly, since random splitting
would let the model "recognize" a domain it already saw during training.

## Metrics (real PhiUSIIL test split, seed 42, `python -m phishnet real-train`)

| Model | Precision | Recall | F1 | ROC-AUC | Recall@1%FPR | Recall@5%FPR |
|---|---|---|---|---|---|---|
| Attention NN (char, URL-only) | 0.901 | 0.545 | 0.679 | 0.819 | 0.396 | 0.526 |
| TF-IDF char n-gram + LogReg (URL-only, **deployed**) | 1.000 | 0.992 | 0.996 | 0.998 | 0.996 | 0.997 |
| Gradient Boosting (URL-only) | 0.999 | 0.995 | 0.997 | 0.999 | 0.996 | 0.997 |
| Gradient Boosting (full features, upper bound) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The near-perfect URL-only scores are not leakage in the circular sense
(no single URL-only feature exceeds the 0.98 single-feature-AUC
suspicious threshold), but they are inflated by a real dataset-compilation
artifact: 100% of PhiUSIIL's legitimate URLs use HTTPS, which is not true
of real-world traffic and is an eroding signal as phishing increasingly
uses free automated certificates. See README's Leakage Controls and
Limitations sections for the full investigation.

## Not recommended for

Any real phishing-blocking decision without further validation on
traffic outside this specific dataset's collection pipeline. The
`IsHTTPS`-driven separability documented above is dataset-specific and
should not be assumed to transfer to production traffic where legitimate
sites without HTTPS and phishing sites with free HTTPS certificates are
both common. See `docs/threat_model.md` for the full evasion discussion.
