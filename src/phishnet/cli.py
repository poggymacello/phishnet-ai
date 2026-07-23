"""Command-line entry point: ``python -m phishnet train|eval``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phishnet import baseline as baseline_mod
from phishnet import evaluate as eval_mod
from phishnet import model as model_mod
from phishnet.artifact import artifact_path_for_version, build_artifact, save_artifact
from phishnet.data import Vocabulary, generate_dataset, train_val_test_split
from phishnet.real_pipeline import run_full_pipeline


def _run_pipeline(n_samples: int, epochs: int, seed: int) -> dict:
    dataset = generate_dataset(n_samples=n_samples, seed=seed)
    train, val, test = train_val_test_split(dataset, seed=seed)

    vocab = Vocabulary(max_len=12, max_vocab=200).fit(train.emails)
    X_train = vocab.encode(train.emails)
    X_test = vocab.encode(test.emails)

    nn_model = model_mod.AttentionClassifier(vocab_size=vocab.size)
    loss_history = model_mod.train_model(nn_model, X_train, train.labels, epochs=epochs, seed=seed)
    nn_probs = model_mod.predict_proba(nn_model, X_test)
    nn_metrics = eval_mod.compute_metrics(test.labels, nn_probs)

    baseline_pipeline = baseline_mod.train_baseline(train.emails, train.labels, seed=seed)
    baseline_probs = baseline_mod.predict_proba(baseline_pipeline, test.emails)
    baseline_metrics = eval_mod.compute_metrics(test.labels, baseline_probs)

    phishing_idx = [i for i, label in enumerate(test.labels) if label == 1]
    sample_idx = phishing_idx[: min(20, len(phishing_idx))]
    sample_X = X_test[sample_idx]
    attn = model_mod.attention_weights(nn_model, sample_X)
    token_lists = [vocab.tokens(test.emails[i]) for i in sample_idx]
    trigger_stats = eval_mod.attention_trigger_score(attn, token_lists)

    return {
        "vocab": vocab,
        "test": test,
        "loss_history": loss_history,
        "nn_probs": nn_probs,
        "nn_metrics": nn_metrics,
        "baseline_probs": baseline_probs,
        "baseline_metrics": baseline_metrics,
        "sample_attn": attn,
        "sample_tokens": token_lists,
        "trigger_stats": trigger_stats,
    }


def cmd_train(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = _run_pipeline(args.n_samples, args.epochs, args.seed)
    test = result["test"]

    metrics = {
        "attention_model": result["nn_metrics"].as_dict(),
        "tfidf_logreg_baseline": result["baseline_metrics"].as_dict(),
        "attention_trigger_check": result["trigger_stats"],
        "final_train_loss": round(result["loss_history"][-1], 4),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    eval_mod.plot_confusion_matrix(
        result["nn_metrics"].confusion, out_dir / "confusion_matrix.png", "attention model"
    )
    eval_mod.plot_roc_curves(
        {
            "attention model": (test.labels, result["nn_probs"]),
            "TF-IDF + logistic regression": (test.labels, result["baseline_probs"]),
        },
        out_dir / "roc_curve.png",
    )
    if result["sample_attn"].shape[0] > 0:
        eval_mod.plot_attention_heatmap(result["sample_attn"][0], out_dir / "heatmap_attention.png")

    print(json.dumps(metrics, indent=2))
    print(f"\nartifacts written to {out_dir}/")


def cmd_eval(args: argparse.Namespace) -> None:
    result = _run_pipeline(args.n_samples, args.epochs, args.seed)
    print("attention model:", result["nn_metrics"].as_dict())
    print("baseline:       ", result["baseline_metrics"].as_dict())
    print("attention trigger check:", result["trigger_stats"])


def cmd_real_train(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_full_pipeline(seed=args.seed)
    test = result["test"]

    report = {
        "metrics": result["metrics"],
        "polarity_flags": result["polarity_flags"],
        "trigger_stats": result["trigger_stats"],
        "base_rates": result["base_rates"],
        "leakage_audit": result["leakage_audit"],
        "nn_fit_seconds": round(result["nn_fit_seconds"], 2),
    }
    (out_dir / "metrics_real.json").write_text(json.dumps(report, indent=2))

    roc_inputs = {name: (test.labels, scores) for name, scores in result["scores"].items()}
    eval_mod.plot_roc_curves(roc_inputs, out_dir / "roc_curve_real.png")

    print(json.dumps(report, indent=2))
    print(f"\nartifacts written to {out_dir}/")

    print("\ntraining and saving the deployed TF-IDF artifact...")
    train_full = result["train_full"]
    artifact = build_artifact(list(train_full.urls), list(train_full.labels), seed=args.seed)
    path = artifact_path_for_version(Path(args.models_dir))
    save_artifact(artifact, path)
    print(f"model artifact written to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phishnet")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--n-samples", type=int, default=600)
    common.add_argument("--epochs", type=int, default=30)
    common.add_argument("--seed", type=int, default=42)

    train_p = sub.add_parser("train", parents=[common], help="train + evaluate + save artifacts")
    train_p.add_argument("--output-dir", default="assets")
    train_p.set_defaults(func=cmd_train)

    eval_p = sub.add_parser(
        "eval", parents=[common], help="re-run the deterministic pipeline and print metrics"
    )
    eval_p.set_defaults(func=cmd_eval)

    real_train_p = sub.add_parser(
        "real-train",
        help="train + evaluate on the real PhiUSIIL data and save the deployment artifact",
    )
    real_train_p.add_argument("--seed", type=int, default=42)
    real_train_p.add_argument("--output-dir", default="assets")
    real_train_p.add_argument("--models-dir", default="models")
    real_train_p.set_defaults(func=cmd_real_train)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
