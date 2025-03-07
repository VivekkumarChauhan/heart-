import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask App
app = Flask(__name__)

# Load trained model and preprocessing tools
model = joblib.load("heart.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Manually define expected input features (must match training data)
FEATURES = [
    'BMICategory', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 
    'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 
    'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer'
]


# Route for home
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Heart Disease Prediction API!"})

# Function to safely encode categorical values
def encode_input_data(input_df):
    for col in label_encoders:
        if col in input_df:
            encoder = label_encoders[col]
            # Replace unseen values with the most frequent category
            input_df[col] = input_df[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
            # Apply label encoding
            input_df[col] = encoder.transform(input_df[col])
    return input_df

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data as JSON
        data = request.get_json()

        # Check if all required features are present
        missing_features = [feature for feature in FEATURES if feature not in data]
        if missing_features:
            return jsonify({"error": f"Missing input features: {missing_features}"}), 400

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([data])

        # Reorder columns to match training order
        input_df = input_df[FEATURES]

        # Encode categorical values safely
        input_df = encode_input_data(input_df)

        # Scale the input data
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_text = "Yes" if prediction == 1 else "No"

        return jsonify({"Heart Disease Prediction": prediction_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
