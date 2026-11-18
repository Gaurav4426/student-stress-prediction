from flask import Flask, render_template, request
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

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    message = None

    if request.method == "POST":
        try:
            data = {
                "prev_avg_hrv": float(request.form["prev_avg_hrv"]),
                "prev_hrv_coverage": float(request.form["prev_hrv_coverage"]),
                "prev_avg_lf": float(request.form["prev_avg_lf"]),
                "prev_avg_hf": float(request.form["prev_avg_hf"]),
                "prev_sleep_score": float(request.form["prev_sleep_score"]),
                "prev_deep_sleep_min": float(request.form["prev_deep_sleep_min"]),
                "prev_total_steps": float(request.form["prev_total_steps"]),
                "prev_avg_steps_interval": float(request.form["prev_avg_steps_interval"]),
                "prev_avg_oxygen": float(request.form["prev_avg_oxygen"]),
                "prev_median_oxygen": float(request.form["prev_median_oxygen"]),
                "prev_LIGHTLY_ACTIVE": float(request.form["prev_LIGHTLY_ACTIVE"]),
                "prev_MODERATELY_ACTIVE": float(request.form["prev_MODERATELY_ACTIVE"]),
                "prev_SEDENTARY": float(request.form["prev_SEDENTARY"]),
                "prev_VERY_ACTIVE": float(request.form["prev_VERY_ACTIVE"]),
                "prev_stress": float(request.form["prev_stress"]),
                "prev_anxiety": float(request.form["prev_anxiety"]),
                "gap_days": float(request.form["gap_days"]),
                "prev_weekday": request.form["prev_weekday"]
            }

            input_data = pd.DataFrame([data], columns=features)
            result = model.predict(input_data)[0]

            if result == 1:
                prediction = "High Stress Risk"
            else:
                prediction = "Low Stress Risk"

        except Exception as e:
            message = "Please enter valid values."

    return render_template(
        "index.html",
        prediction=prediction,
        message=message
    )

if __name__ == "__main__":
    app.run()