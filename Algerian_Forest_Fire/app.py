import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Absolute path to the Algerian_Forest_Fire/ folder (where app.py lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load model & scaler ──────────────────────────────────────────────────────
model  = pickle.load(open(os.path.join(BASE_DIR, "models", "ridge_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "models", "scaler.pkl"),      "rb"))

# FWI is the TARGET (what we predict) — scaler expects only these 9 input features
FEATURES = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI"]

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values       = [float(request.form[f]) for f in FEATURES]
        input_array  = np.array(values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)        # ← scaler applied here
        prediction   = model.predict(input_scaled)[0]

        fire_class = "🔥 Fire Risk"    if prediction > 0 else "✅ No Fire Risk"
        fire_flag  = prediction > 0

        return render_template(
            "index.html",
            prediction_text=f"Predicted FWI: {prediction:.2f}  →  {fire_class}",
            fire=fire_flag,
            input_values=dict(zip(FEATURES, values))
        )
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}", fire=False)


@app.route("/metrics")
def metrics():
    metrics_data = {
        "model"   : "Ridge Regression",
        "r2_score": 0.98,    # ← replace with your real value from notebook
        "mae"     : 0.45,    # ← replace with your real value
        "mse"     : 0.31,    # ← replace with your real value
        "rmse"    : 0.56,    # ← replace with your real value
    }
    return render_template("metrics.html", metrics=metrics_data)


@app.route("/eda")
def eda():
    csv_path    = os.path.join(BASE_DIR, "data", "processed_data", "processed_algerian_forest_fire_data.csv")
    df          = pd.read_csv(csv_path)
    stats       = df.describe().round(2).to_html(classes="stats-table", border=0)
    fire_counts = df["Classes"].value_counts().to_dict()
    return render_template("eda.html", stats=stats, fire_counts=fire_counts)


if __name__ == "__main__":
    app.run(debug=True)
