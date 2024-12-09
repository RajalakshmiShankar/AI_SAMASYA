from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('autism_detector_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect inputs for 10 questions
        features = [int(request.form[f'A{i}']) for i in range(1, 11)]
    except ValueError:
        return render_template('index.html', error="Please enter valid inputs (0 or 1) for all questions.")

    # Convert features to numpy array
    features = np.array([features])

    # Get prediction probabilities
    probabilities = model.predict_proba(features)[0]

    # Debugging: Print the probabilities to console
    print("Prediction Probabilities:", probabilities)

    # Class prediction based on threshold (0.5 default)
    if probabilities[1] >= 0.5:
        result = "Positive for Autism Traits"
    else:
        result = "Negative for Autism Traits"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)


