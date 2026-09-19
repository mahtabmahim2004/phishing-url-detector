(function () {
    "use strict";

    const form = document.getElementById("prediction-form");
    const input = document.getElementById("url-input");
    const button = document.getElementById("analyze-button");
    const loadingMessage = document.getElementById("loading-message");
    const resultCard = document.getElementById("result-card");
    const errorMessage = document.getElementById("error-message");
    const predictionValue = document.getElementById("prediction-value");
    const labelValue = document.getElementById("label-value");
    const probabilityValue = document.getElementById("probability-value");

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.hidden = false;
        resultCard.hidden = true;
    }

    function showResult(data) {
        predictionValue.textContent = data.prediction === "phishing" ? "Phishing" : "Benign";
        labelValue.textContent = String(data.label);
        probabilityValue.textContent = typeof data.probability === "number"
            ? `${(data.probability * 100).toFixed(1)}%`
            : "Unavailable";
        resultCard.hidden = false;
        errorMessage.hidden = true;
    }

    form.addEventListener("submit", async function (event) {
        event.preventDefault();
        const url = input.value.trim();
        if (!url) {
            showError("Enter a URL to analyze.");
            input.focus();
            return;
        }

        button.disabled = true;
        loadingMessage.hidden = false;
        resultCard.hidden = true;
        errorMessage.hidden = true;

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url: url })
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || "The URL could not be analyzed.");
            }
            showResult(data);
        } catch (error) {
            showError(error.message || "The URL could not be analyzed.");
        } finally {
            button.disabled = false;
            loadingMessage.hidden = true;
        }
    });
})();