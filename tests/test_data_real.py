from pathlib import Path

import pandas as pd

from phishnet.data_real import CharVocabulary, group_split, prepare

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "phiusiil_sample.csv"


def _load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


def test_prepare_flips_label_convention():
    df = _load_sample()
    dataset = prepare(df)
    # PhiUSIIL: label=1 means legitimate. This project's convention is
    # 1 = phishing, so a raw-label-1 row must come out as 0 here.
    raw_legit_mask = df["label"] == 1
    assert (dataset.labels[raw_legit_mask.to_numpy()] == 0).all()
    raw_phish_mask = df["label"] == 0
    assert (dataset.labels[raw_phish_mask.to_numpy()] == 1).all()


def test_prepare_no_nan_or_inf():
    dataset = prepare(_load_sample())
    assert not dataset.url_only_features.isna().any().any()
    assert not dataset.page_content_features.isna().any().any()


def test_group_split_no_domain_overlap():
    dataset = prepare(_load_sample())
    train, val, test = group_split(dataset, seed=1)

    train_domains = set(train.etld1)
    val_domains = set(val.etld1)
    test_domains = set(test.etld1)

    assert train_domains.isdisjoint(val_domains)
    assert train_domains.isdisjoint(test_domains)
    assert val_domains.isdisjoint(test_domains)
    assert len(train) + len(val) + len(test) == len(dataset)


def test_group_split_deterministic_for_same_seed():
    dataset = prepare(_load_sample())
    train_a, _, _ = group_split(dataset, seed=7)
    train_b, _, _ = group_split(dataset, seed=7)
    assert list(train_a.urls) == list(train_b.urls)


def test_char_vocabulary_fit_only_on_training_urls():
    dataset = prepare(_load_sample())
    train, _, test = group_split(dataset, seed=1)

    vocab = CharVocabulary(max_len=40).fit(train.urls)
    encoded = vocab.encode(test.urls)
    assert encoded.shape == (len(test), 40)
    # padding index 0 must never collide with a real fitted character index
    assert 0 not in vocab.char_to_idx.values()
