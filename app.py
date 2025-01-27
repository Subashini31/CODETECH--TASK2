from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import json

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('trained_model.pkl')

# Define a route for the root URL
@app.route('/')
def home():
    return render_template('heart.html')  # Load the HTML file

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the POST request
        data = request.form  # Input from HTML form
        
        # Extract and convert features to float
        features = [
            float(data['age']),
            float(data['sex']),
            float(data['chest_pain_type']),
            float(data['resting_blood_pressure']),
            float(data['serum_cholesterol']),
            float(data['fasting_blood_sugar']),
            float(data['resting_ecg']),
            float(data['max_heart_rate']),
            float(data['exercise_induced_angina']),
            float(data['oldpeak']),
            float(data['slope']),
            float(data['num_major_vessels']),
            float(data['thal'])
        ]

        # Convert features into a numpy array for prediction
        features = np.array(features).reshape(1, -1)

        # Make prediction using the model
        prediction = model.predict(features)

        # Return the result
        return render_template('heart.html', prediction_text=f"Heart Disease Prediction: {'Disease' if int(prediction[0]) == 1 else 'No Disease'}")
    except Exception as e:
        return render_template('heart.html', error_text=f"Error: {str(e)}")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
