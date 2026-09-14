import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"

with MODEL_PATH.open("rb") as file:
    model = pickle.load(file)

# Assignment requirement 7: Model saving & API endpoint setup.
# These features align with the model input contract used during training.
FEATURE_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


# Assignment requirement 3: Feature engineering is applied before prediction.
# This keeps the API preprocessing consistent with the training workflow.
def add_engineered_features(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["tenure_to_monthly_ratio"] = df["tenure"] / (df["MonthlyCharges"] + 1e-6)

    addon_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["num_addon_services"] = (df[addon_cols] == "Yes").sum(axis=1)
    return df


# Assignment requirement 7: REST API implementation.
# Endpoint: POST /predict
# Steps: validate input, preprocess the record, load trained model, and return prediction + probability.
@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "Request JSON body is required."}), 400

    missing = [column for column in FEATURE_COLUMNS if column not in payload and column != "customerID"]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    record = {column: payload.get(column) for column in FEATURE_COLUMNS if column in payload}
    record = add_engineered_features(pd.DataFrame([record]))

    prediction = model.predict(record)[0]
    churn_probability = float(model.predict_proba(record)[0][1])

    return jsonify({
        "prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(churn_probability, 4),
    })



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
