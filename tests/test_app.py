import app as app_module
from feature_extraction import FEATURE_NAMES


class FakeClassifier:
	classes_ = [0, 1]

	def predict(self, features):
		return [0]

	def predict_proba(self, features):
		return [[0.8, 0.2]]


def create_test_client():
	model_bundle = {
		"model": FakeClassifier(),
		"feature_columns": list(FEATURE_NAMES),
	}
	return app_module.create_app(model_bundle=model_bundle).test_client()


def test_index_page_loads():
	client = create_test_client()

	response = client.get("/")
	script_response = client.get("/static/js/app.js")

	assert response.status_code == 200
	assert b"Phishing URL Detector" in response.data
	assert b"/static/js/app.js" in response.data
	assert script_response.status_code == 200
	assert b'fetch("/api/predict"' in script_response.data


def test_health_endpoint_with_model():
	client = create_test_client()

	response = client.get("/api/health")

	assert response.status_code == 200
	assert response.get_json() == {"status": "ok"}


def test_valid_prediction():
	client = create_test_client()

	response = client.post("/api/predict", json={"url": "https://example.com"})

	assert response.status_code == 200
	assert response.get_json() == {
		"url": "https://example.com",
		"label": 0,
		"prediction": "benign",
		"probability": 0.8,
	}


def test_missing_json_body():
	client = create_test_client()

	response = client.post("/api/predict")

	assert response.status_code == 400


def test_missing_url():
	client = create_test_client()

	response = client.post("/api/predict", json={})

	assert response.status_code == 400


def test_empty_url():
	client = create_test_client()

	response = client.post("/api/predict", json={"url": "   "})

	assert response.status_code == 400


def test_non_string_url():
	client = create_test_client()

	response = client.post("/api/predict", json={"url": 123})

	assert response.status_code == 400


def test_oversized_url():
	client = create_test_client()

	response = client.post("/api/predict", json={"url": "a" * 4097})

	assert response.status_code == 413


def test_model_unavailable():
	client = app_module.create_app(model_path="missing-model.joblib").test_client()

	health_response = client.get("/api/health")
	predict_response = client.post("/api/predict", json={"url": "https://example.com"})

	assert health_response.status_code == 503
	assert health_response.get_json() == {"status": "error", "error": "model unavailable"}
	assert predict_response.status_code == 503