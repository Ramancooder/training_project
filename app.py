from datetime import datetime
from pathlib import Path
from threading import Timer
import webbrowser

import joblib
import pandas as pd
from flask import Flask, render_template, request

from features import extract_features, parse_url


app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"

if not MODEL_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
    raise FileNotFoundError(
        "Model files are missing. Run train_model.py successfully before starting Flask."
    )

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
scan_history = []

FEATURE_DISPLAY_NAMES = {
    "url_length": "URL Length",
    "domain_length": "Domain Length",
    "path_length": "Path Length",
    "query_length": "Query Length",
    "num_dots": "Number of Dots",
    "num_slashes": "Number of Slashes",
    "num_hyphens": "Number of Hyphens",
    "num_digits": "Number of Digits",
    "num_letters": "Number of Letters",
    "special_character_count": "Special Characters",
    "number_of_subdomains": "Subdomains (heuristic)",
    "num_subdirectories": "Subdirectories",
    "digit_ratio": "Digit Ratio",
    "letter_ratio": "Letter Ratio",
    "has_nonstandard_port": "Nonstandard Port",
}


def probability_for_class(probabilities, class_label):
    classes = list(model.classes_)
    try:
        return float(probabilities[classes.index(class_label)]) * 100.0
    except (ValueError, IndexError):
        return 0.0


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    risk_score = None
    risk_level = None
    url = ""
    extracted_features = {}
    security_checks = []
    feature_analysis = []
    error = None

    if request.method == "POST":
        url = request.form.get("url", "").strip()

        if not url:
            error = "Please enter a URL."
        else:
            try:
                normalized_url, parsed_url, hostname, port = parse_url(url)
                extracted_features = extract_features(url)

                missing = [column for column in feature_columns if column not in extracted_features]
                extra = [column for column in extracted_features if column not in feature_columns]
                if missing or extra:
                    raise RuntimeError(
                        "Feature mismatch between the loaded model and features.py. "
                        "Run train_model.py again. "
                        f"Missing: {missing}; extra: {extra}"
                    )

                input_data = pd.DataFrame(
                    [[extracted_features[column] for column in feature_columns]],
                    columns=feature_columns,
                )

                prediction = int(model.predict(input_data)[0])
                probabilities = model.predict_proba(input_data)[0]
                phishing_probability = probability_for_class(probabilities, 1)
                legitimate_probability = probability_for_class(probabilities, 0)

                if prediction == 1:
                    result = "Phishing"
                    confidence = phishing_probability
                else:
                    result = "Legitimate"
                    confidence = legitimate_probability

                # This is the model's estimated phishing probability, not a guarantee.
                risk_score = round(phishing_probability, 2)
                confidence = round(confidence, 2)

                security_checks = [
                    {
                        "name": "HTTPS",
                        "safe": extracted_features["https"] == 1,
                        "value": "Enabled" if extracted_features["https"] == 1 else "Not detected",
                    },
                    {
                        "name": "IP Address",
                        "safe": extracted_features["has_ip"] == 0,
                        "value": "Not detected" if extracted_features["has_ip"] == 0 else "Detected",
                    },
                    {
                        "name": "Suspicious Words",
                        "safe": extracted_features["has_suspicious_words"] == 0,
                        "value": "Not detected" if extracted_features["has_suspicious_words"] == 0 else "Detected",
                    },
                    {
                        "name": "Suspicious Domain Words",
                        "safe": extracted_features["suspicious_word_in_domain"] == 0,
                        "value": "Not detected" if extracted_features["suspicious_word_in_domain"] == 0 else "Detected",
                    },
                    {
                        "name": "URL Shortener",
                        "safe": extracted_features["has_shortened_url"] == 0,
                        "value": "Not detected" if extracted_features["has_shortened_url"] == 0 else "Detected",
                    },
                    {
                        "name": "@ Symbol",
                        "safe": extracted_features["has_at_symbol"] == 0,
                        "value": "Not detected" if extracted_features["has_at_symbol"] == 0 else "Detected",
                    },
                    {
                        "name": "Nonstandard Port",
                        "safe": extracted_features["has_nonstandard_port"] == 0,
                        "value": f"Port {port} detected" if extracted_features["has_nonstandard_port"] else "No nonstandard port detected",
                    },
                ]

                analysis_features = [
                    "url_length", "domain_length", "path_length", "query_length",
                    "num_dots", "num_slashes", "num_digits", "num_hyphens",
                    "special_character_count", "number_of_subdomains",
                    "has_nonstandard_port",
                ]
                max_values = {
                    "url_length": 200, "domain_length": 80, "path_length": 100,
                    "query_length": 100, "num_dots": 10, "num_slashes": 10,
                    "num_digits": 30, "num_hyphens": 10,
                    "special_character_count": 50, "number_of_subdomains": 5,
                    "has_nonstandard_port": 1,
                }

                for feature in analysis_features:
                    value = float(extracted_features[feature])
                    percentage = min((value / max_values[feature]) * 100, 100)
                    feature_analysis.append({
                        "name": FEATURE_DISPLAY_NAMES.get(feature, feature.replace("_", " ").title()),
                        "value": round(value, 3),
                        "percentage": round(percentage, 1),
                    })

                scan_history.insert(0, {
                    "url": url,
                    "result": result,
                    "confidence": confidence,
                    "risk": risk_score,
                    "time": datetime.now().strftime("%H:%M:%S"),
                })
                del scan_history[5:]

            except ValueError as exc:
                error = str(exc)
            except Exception as exc:
                error = "Unable to analyze this URL: " + str(exc)
                app.logger.exception("Prediction error")

    if risk_score is not None:
        if risk_score < 25:
            risk_level = "Low"
        elif risk_score < 60:
            risk_level = "Medium"
        else:
            risk_level = "High"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        risk_score=risk_score,
        risk_level=risk_level,
        url=url,
        security_checks=security_checks,
        feature_analysis=feature_analysis,
        scan_history=scan_history,
        feature_count=len(feature_columns),
        error=error,
    )


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")


if __name__ == "__main__":
    app.run(debug=True)
