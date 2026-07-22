from phishnet.data import Vocabulary, generate_dataset, train_val_test_split


def test_generate_dataset_labels_and_size():
    dataset = generate_dataset(n_samples=200, phishing_ratio=0.35, seed=1)
    assert len(dataset.emails) == 200
    assert len(dataset.labels) == 200
    assert set(dataset.labels) == {0, 1}
    phishing_count = sum(dataset.labels)
    assert 50 <= phishing_count <= 90  # roughly 35% of 200, allow generation slack


def test_split_has_no_overlap_and_preserves_total():
    dataset = generate_dataset(n_samples=300, seed=2)
    train, val, test = train_val_test_split(dataset, val_size=0.15, test_size=0.15, seed=2)

    assert len(train.emails) + len(val.emails) + len(test.emails) == 300

    # generate_dataset guarantees every email string is unique, so a
    # correct split must have zero exact-text overlap between train and test
    train_set = set(train.emails)
    test_set = set(test.emails)
    assert train_set.isdisjoint(test_set)


def test_vocabulary_fit_and_encode_shape():
    dataset = generate_dataset(n_samples=100, seed=3)
    vocab = Vocabulary(max_len=12, max_vocab=50).fit(dataset.emails)
    encoded = vocab.encode(dataset.emails)
    assert encoded.shape == (100, 12)
    assert encoded.max() <= vocab.size - 1
    assert encoded.min() >= 0
