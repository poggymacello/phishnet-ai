from phishnet.baseline import predict_proba, train_baseline
from phishnet.data import generate_dataset, train_val_test_split
from phishnet.evaluate import compute_metrics


def test_baseline_beats_chance_on_synthetic_data():
    dataset = generate_dataset(n_samples=400, seed=5)
    train, _val, test = train_val_test_split(dataset, seed=5)

    pipeline = train_baseline(train.emails, train.labels, seed=5)
    probs = predict_proba(pipeline, test.emails)
    metrics = compute_metrics(test.labels, probs)

    # the synthetic templates are lexically very separable, so a working
    # TF-IDF + logistic regression baseline should do well above chance
    assert metrics.f1 > 0.8
    assert metrics.roc_auc > 0.8
