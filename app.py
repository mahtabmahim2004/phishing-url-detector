"""Minimal Flask API for local phishing-URL predictions."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from train_model import DEFAULT_MODEL_PATH, load_model, predict_url


MAX_URL_LENGTH = 4096


def create_app(model_path: str | Path = DEFAULT_MODEL_PATH, model_bundle: dict | None = None) -> Flask:
	"""Create the API without training a model or accessing any submitted URL."""
	app = Flask(__name__)
	if model_bundle is not None:
		app.config["MODEL_BUNDLE"] = model_bundle
		app.config["MODEL_ERROR"] = None
	else:
		try:
			app.config["MODEL_BUNDLE"] = load_model(model_path)
			app.config["MODEL_ERROR"] = None
		except Exception:
			app.config["MODEL_BUNDLE"] = None
			app.config["MODEL_ERROR"] = "model unavailable"

	@app.get("/")
	def index():
		return render_template("index.html")

	@app.get("/api/health")
	def health():
		if app.config["MODEL_BUNDLE"] is None:
			return jsonify({"status": "error", "error": "model unavailable"}), 503
		return jsonify({"status": "ok"})

	@app.post("/api/predict")
	def predict():
		payload = request.get_json(silent=True)
		if not isinstance(payload, dict):
			return jsonify({"error": "request body must be a JSON object"}), 400

		url = payload.get("url")
		if not isinstance(url, str):
			return jsonify({"error": "url must be a string"}), 400
		url = url.strip()
		if not url:
			return jsonify({"error": "url must not be empty"}), 400
		if len(url) > MAX_URL_LENGTH:
			return jsonify({"error": "url exceeds the maximum length"}), 413

		model_bundle = app.config["MODEL_BUNDLE"]
		if model_bundle is None:
			return jsonify({"error": "model unavailable"}), 503

		try:
			label, probability = predict_url(url, model_bundle)
		except Exception:
			return jsonify({"error": "prediction failed"}), 500

		response = {
			"url": url,
			"label": label,
			"prediction": "phishing" if label == 1 else "benign",
			"probability": probability,
		}
		return jsonify(response)

	return app


app = create_app()


if __name__ == "__main__":
	app.run(host="127.0.0.1", port=5000, debug=False)
