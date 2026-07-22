import json

from phishnet.cli import main


def test_train_command_writes_artifacts(tmp_path):
    out_dir = tmp_path / "assets"
    main(
        [
            "train",
            "--n-samples",
            "80",
            "--epochs",
            "3",
            "--seed",
            "1",
            "--output-dir",
            str(out_dir),
        ]
    )

    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "confusion_matrix.png").exists()
    assert (out_dir / "roc_curve.png").exists()
    assert (out_dir / "heatmap_attention.png").exists()

    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert "attention_model" in metrics
    assert "tfidf_logreg_baseline" in metrics
    assert "f1" in metrics["attention_model"]


def test_eval_command_runs_without_error(capsys):
    main(["eval", "--n-samples", "80", "--epochs", "3", "--seed", "1"])
    captured = capsys.readouterr()
    assert "attention model" in captured.out
