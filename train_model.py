"""URL dataset preparation and the Phase 4 model training workflow.

All URL handling remains local and string-based. URLs are never visited,
resolved, or otherwise accessed over the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from feature_extraction import FEATURE_NAMES, extract_features_batch


DEFAULT_DATASET_PATH = Path(__file__).parent / "data" / "raw" / "urls.csv"
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "phishing_url_model.joblib"
REQUIRED_COLUMNS = {"url", "label"}


class DatasetValidationError(ValueError):
	"""Raised when a URL dataset does not match the expected format."""


@dataclass(frozen=True)
class DataCleaningReport:
	"""Counts and class information produced while cleaning a dataset."""

	input_records: int
	removed_records: int
	output_records: int
	class_distribution: dict[int, int]


def _normalize_labels(labels: pd.Series) -> pd.Series:
	normalized = pd.to_numeric(labels.astype("string").str.strip(), errors="coerce")
	invalid = normalized.isna() | ~normalized.isin([0, 1])
	if invalid.any():
		invalid_values = labels[invalid].drop_duplicates().tolist()
		raise DatasetValidationError(
			"Labels must contain only 0 (benign) or 1 (phishing); "
			f"invalid values: {invalid_values}"
		)
	return normalized.astype("int64")


def preprocess_url_data(data: pd.DataFrame) -> tuple[pd.DataFrame, DataCleaningReport]:
	"""Clean URL records and return the cleaned frame plus a report.

	URLs are treated as ordinary strings. No network operation is performed.
	Missing, blank, and duplicate URL records are removed. Missing labels are
	removed, while non-missing labels outside 0 and 1 are rejected.
	"""
	missing_columns = REQUIRED_COLUMNS - set(data.columns)
	if missing_columns:
		names = ", ".join(sorted(missing_columns))
		raise DatasetValidationError(f"Dataset is missing required column(s): {names}")

	records = data[["url", "label"]].copy()
	input_records = len(records)
	records["url"] = records["url"].astype("string").str.strip()
	records = records.dropna(subset=["url", "label"])
	records = records[records["url"] != ""]
	records["label"] = _normalize_labels(records["label"])
	records = records.drop_duplicates(subset=["url"], keep="first")
	records = records.reset_index(drop=True)

	report = DataCleaningReport(
		input_records=input_records,
		removed_records=input_records - len(records),
		output_records=len(records),
		class_distribution={
			int(label): int(count)
			for label, count in records["label"].value_counts().sort_index().items()
		},
	)
	return records, report


def load_url_dataset(
	path: Union[str, Path] = DEFAULT_DATASET_PATH,
	*,
	return_report: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, DataCleaningReport]:
	"""Read, validate, and clean a URL CSV dataset using pandas."""
	dataset_path = Path(path)
	if not dataset_path.exists():
		raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

	try:
		data = pd.read_csv(dataset_path)
	except (OSError, pd.errors.ParserError) as exc:
		raise DatasetValidationError(f"Could not read dataset {dataset_path}: {exc}") from exc

	records, report = preprocess_url_data(data)
	if report.output_records == 0:
		raise DatasetValidationError("Dataset contains no valid URL records after preprocessing")
	if return_report:
		return records, report
	return records


def prepare_train_test_split(
	data: pd.DataFrame,
	*,
	test_size: float = 0.2,
	random_state: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
	"""Create deterministic train/test URL and label sets without leakage."""
	records, _ = preprocess_url_data(data)
	if len(records) < 2:
		raise DatasetValidationError("At least two valid URL records are required for a split")

	test_count = max(1, int(len(records) * test_size + 0.999999))
	train_count = len(records) - test_count
	class_count = records["label"].nunique()
	can_stratify = (
		class_count > 1
		and records["label"].value_counts().min() >= 2
		and test_count >= class_count
		and train_count >= class_count
	)
	stratify = records["label"] if can_stratify else None

	return train_test_split(
		records["url"],
		records["label"],
		test_size=test_size,
		random_state=random_state,
		stratify=stratify,
	)


def train_model(
	dataset_path: Union[str, Path] = DEFAULT_DATASET_PATH,
	model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
	*,
	test_size: float = 0.2,
	random_state: int = 42,
) -> dict:
	"""Train and save a small deterministic Random Forest model.

	Returns evaluation details and the saved model bundle. Metrics describe the
	provided dataset only and are not a measure of real-world performance.
	"""
	records, data_report = load_url_dataset(dataset_path, return_report=True)
	train_urls, test_urls, y_train, y_test = prepare_train_test_split(
		records,
		test_size=test_size,
		random_state=random_state,
	)

	X_train = extract_features_batch(train_urls)
	X_test = extract_features_batch(test_urls)
	classifier = RandomForestClassifier(
		n_estimators=100,
		max_depth=8,
		random_state=random_state,
		n_jobs=1,
	)
	classifier.fit(X_train, y_train)
	predictions = classifier.predict(X_test)

	metrics = {
		"accuracy": float(accuracy_score(y_test, predictions)),
		"precision": float(precision_score(y_test, predictions, zero_division=0)),
		"recall": float(recall_score(y_test, predictions, zero_division=0)),
		"f1": float(f1_score(y_test, predictions, zero_division=0)),
	}
	result = {
		"metrics": metrics,
		"confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
		"dataset_report": data_report,
		"test_records": len(y_test),
	}

	model_bundle = {"model": classifier, "feature_columns": list(FEATURE_NAMES)}
	artifact_path = Path(model_path)
	artifact_path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(model_bundle, artifact_path)
	result["model_bundle"] = model_bundle
	result["model_path"] = artifact_path
	return result


def load_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> dict:
	"""Load and validate a saved model bundle."""
	bundle = joblib.load(Path(model_path))
	if not isinstance(bundle, dict) or "model" not in bundle or "feature_columns" not in bundle:
		raise ValueError("Model artifact must contain 'model' and 'feature_columns'")
	if tuple(bundle["feature_columns"]) != FEATURE_NAMES:
		raise ValueError("Model artifact feature columns do not match the extractor")
	if not hasattr(bundle["model"], "predict"):
		raise ValueError("Model artifact does not contain a usable classifier")
	return bundle


def predict_url(url: str, model_bundle: dict) -> tuple[int, float | None]:
	"""Predict one URL using a validated model bundle without network access."""
	if not isinstance(model_bundle, dict) or "model" not in model_bundle:
		raise ValueError("Model bundle must contain a model")
	if tuple(model_bundle.get("feature_columns", ())) != FEATURE_NAMES:
		raise ValueError("Model bundle feature columns do not match the extractor")

	features = extract_features_batch([url])[list(model_bundle["feature_columns"])]
	classifier = model_bundle["model"]
	predicted_label = int(classifier.predict(features)[0])
	prediction_probability = None
	if hasattr(classifier, "predict_proba") and hasattr(classifier, "classes_"):
		probabilities = classifier.predict_proba(features)[0]
		matching_classes = list(classifier.classes_).index(predicted_label)
		prediction_probability = float(probabilities[matching_classes])
	return predicted_label, prediction_probability


def main() -> None:
	"""Train the development model and print a concise evaluation summary."""
	result = train_model()
	metrics = result["metrics"]
	print(f"Trained Random Forest on {result['dataset_report'].output_records} URL records.")
	print(f"Development test records: {result['test_records']}")
	print(", ".join(f"{name}={value:.3f}" for name, value in metrics.items()))
	print(f"Saved model: {result['model_path']}")
	print("Metrics are for the tiny synthetic development dataset only.")


if __name__ == "__main__":
	main()
