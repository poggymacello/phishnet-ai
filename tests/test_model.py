import numpy as np

from phishnet.data import Vocabulary, generate_dataset
from phishnet.model import AttentionClassifier, attention_weights, predict_proba, train_model


def test_forward_pass_output_shapes():
    model = AttentionClassifier(vocab_size=20)
    X = np.random.randint(0, 20, size=(4, 12))
    probs = predict_proba(model, X)
    attn = attention_weights(model, X)
    assert probs.shape == (4,)
    assert attn.shape == (4, 12, 12)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_training_reduces_loss_on_real_signal():
    dataset = generate_dataset(n_samples=120, seed=0)
    vocab = Vocabulary(max_len=12, max_vocab=100).fit(dataset.emails)
    X = vocab.encode(dataset.emails)

    model = AttentionClassifier(vocab_size=vocab.size)
    history = train_model(model, X, dataset.labels, epochs=30, seed=0)

    assert len(history) == 30
    assert history[-1] < history[0] * 0.5  # meaningful drop, not just noise
