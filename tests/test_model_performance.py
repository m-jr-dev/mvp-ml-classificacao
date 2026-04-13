
from __future__ import annotations

import json
from pathlib import Path

from train_model import train_and_export


def test_selected_model_meets_performance_thresholds():
    metadata = train_and_export()
    selected = metadata["selected_model_metrics"]
    thresholds = metadata["thresholds"]

    assert selected["accuracy"] >= thresholds["minimum_accuracy"]
    assert selected["f1_macro"] >= thresholds["minimum_f1_macro"]


def test_artifacts_are_generated():
    metadata_path = Path("artifacts/model_metadata.json")
    model_path = Path("artifacts/wine_classifier.joblib")

    if not metadata_path.exists() or not model_path.exists():
        train_and_export()

    assert metadata_path.exists()
    assert model_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["selected_model"] in {"KNN", "Árvore de Decisão", "Naive Bayes", "SVM"}
