# Model Card: PhishNet AI

## Purpose

Classifies email text as phishing or normal, comparing a small attention-based
neural network against a TF-IDF + logistic regression baseline. A portfolio
demonstration of the modeling pipeline (attention mechanism, baseline
comparison, interpretability check), not a production phishing filter.

## Data

Entirely synthetic (`src/phishnet/data.py`): 12 phishing templates and 12
normal templates, randomized brand names/links/dates, deduplicated at
generation time. No real phishing corpus. See README's Data and Limitations
sections, and Roadmap for the planned real-data migration.

## Metrics (synthetic test split, seed 42)

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Attention NN | 0.962 | 0.781 | 0.862 | 0.990 |
| TF-IDF + logistic regression | 1.000 | 1.000 | 1.000 | 1.000 |

The baseline's perfect score is a ceiling effect of synthetic data with
disjoint phishing/normal vocabulary, not evidence of a production-ready
classifier — see README for the full discussion.

## Limitations

- Synthetic data only; not validated on real email.
- Attention-interpretability check is a simple average-attention comparison,
  not a rigorous attribution method, and it came back showing no meaningful
  difference between trigger-word and non-trigger-word attention (see README).

## Not recommended for

Any real phishing-detection decision. This is a methodology demonstration,
not a validated filter — see the Roadmap section in the README for what
real-world validation would require.
