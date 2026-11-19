from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")

features = [
    "prev_avg_hrv",
    "prev_hrv_coverage",
    "prev_avg_lf",
    "prev_avg_hf",
    "prev_sleep_score",
    "prev_deep_sleep_min",
    "prev_total_steps",
    "prev_avg_steps_interval",
    "prev_avg_oxygen",
    "prev_median_oxygen",
    "prev_LIGHTLY_ACTIVE",
    "prev_MODERATELY_ACTIVE",
    "prev_SEDENTARY",
    "prev_VERY_ACTIVE",
    "prev_stress",
    "prev_anxiety",
    "gap_days",
    "prev_weekday"
]

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()

        input_data = pd.DataFrame([{
            "prev_avg_hrv": float(data["prev_avg_hrv"]),
            "prev_hrv_coverage": float(data["prev_hrv_coverage"]),
            "prev_avg_lf": float(data["prev_avg_lf"]),
            "prev_avg_hf": float(data["prev_avg_hf"]),
            "prev_sleep_score": float(data["prev_sleep_score"]),
            "prev_deep_sleep_min": float(data["prev_deep_sleep_min"]),
            "prev_total_steps": float(data["prev_total_steps"]),
            "prev_avg_steps_interval": float(data["prev_avg_steps_interval"]),
            "prev_avg_oxygen": float(data["prev_avg_oxygen"]),
            "prev_median_oxygen": float(data["prev_median_oxygen"]),
            "prev_LIGHTLY_ACTIVE": float(data["prev_LIGHTLY_ACTIVE"]),
            "prev_MODERATELY_ACTIVE": float(data["prev_MODERATELY_ACTIVE"]),
            "prev_SEDENTARY": float(data["prev_SEDENTARY"]),
            "prev_VERY_ACTIVE": float(data["prev_VERY_ACTIVE"]),
            "prev_stress": float(data["prev_stress"]),
            "prev_anxiety": float(data["prev_anxiety"]),
            "gap_days": float(data["gap_days"]),
            "prev_weekday": data["prev_weekday"]
        }], columns=features)

        result = model.predict(input_data)[0]

        if result == 1:
            prediction = "High Stress Risk"
        else:
            prediction = "Low Stress Risk"

        return jsonify({
            "success": True,
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run()
