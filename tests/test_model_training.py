from pathlib import Path

from feature_extraction import FEATURE_NAMES
from train_model import load_model, predict_url, train_model


def test_training_saves_valid_model_bundle(tmp_path):
    model_path = tmp_path / "model.joblib"

    result = train_model(model_path=model_path)

    assert model_path.exists()
    assert result["model_bundle"]["feature_columns"] == list(FEATURE_NAMES)
    assert set(result["metrics"]) == {"accuracy", "precision", "recall", "f1"}
    assert len(result["confusion_matrix"]) == 2


def test_saved_model_can_be_loaded(tmp_path):
    model_path = tmp_path / "model.joblib"
    train_model(model_path=model_path)

    bundle = load_model(model_path)

    assert tuple(bundle["feature_columns"]) == FEATURE_NAMES
    assert hasattr(bundle["model"], "predict")


def test_prediction_helper_preserves_feature_order(tmp_path):
    model_path = tmp_path / "model.joblib"
    train_model(model_path=model_path)
    bundle = load_model(model_path)

    label, probability = predict_url("https://example.com/login", bundle)

    assert label in {0, 1}
    assert probability is not None
    assert 0.0 <= probability <= 1.0


def test_training_is_deterministic(tmp_path):
    first = train_model(model_path=tmp_path / "first.joblib")
    second = train_model(model_path=tmp_path / "second.joblib")

    assert first["metrics"] == second["metrics"]
    assert first["confusion_matrix"] == second["confusion_matrix"]
    assert first["model_bundle"]["feature_columns"] == second["model_bundle"]["feature_columns"]