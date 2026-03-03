import joblib
import numpy as np
import pandas as pd

# Load saved model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

print("Model loaded successfully\n")

# Example new patient data
# ⚠ Order must match training data columns exactly

new_patient = {
    "Age": 75,
    "Sex": 1,
    "Chest pain type": 4,
    "BP": 180,
    "Cholesterol": 300,
    "FBS over 120": 1,
    "EKG results": 2,
    "Max HR": 90,
    "Exercise angina": 1,
    "ST depression": 4.0,
    "Slope of ST": 3,
    "Number of vessels fluro": 3,
    "Thallium": 7
}

# Convert to DataFrame
new_df = pd.DataFrame([new_patient])

# Scale
new_scaled = scaler.transform(new_df)

# Predict
prediction = model.predict(new_scaled)[0]
probability = model.predict_proba(new_scaled)[0][1]

print("Prediction:", "Heart Disease" if prediction == 1 else "No Heart Disease")
print("Risk Probability:", round(probability * 100, 2), "%")