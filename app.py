from flask import Flask, render_template, request, redirect, url_for
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
    # Get form data
    inputs = [int(request.form[f'A{i}']) for i in range(1, 11)]
    features = np.array([inputs])
    
    # Make prediction
    prediction = model.predict(features)
    result = "Positive for Autism Traits" if prediction[0] == 1 else "Negative for Autism Traits"
    
    # Redirect to appropriate learning path
    if prediction[0] == 1:
        return redirect(url_for('learning_path', result='positive'))
    else:
        return redirect(url_for('learning_path', result='negative'))

@app.route('/learn/<result>')
def learning_path(result):
    if result == 'positive':
        return render_template('learning_positive.html')
    else:
        return render_template('learning_negative.html')

if __name__ == '__main__':
    app.run(debug=True)


@app.route('/activity/<activity_type>')
def activity(activity_type):
    if activity_type == 'sensory':
        return "<h2>Sensory Integration Exercises</h2><p>Activities that involve touch, sound, and movement.</p>"
    elif activity_type == 'social':
        return "<h2>Social Skills Training</h2><p>Role-playing games and interactive exercises.</p>"
    elif activity_type == 'language':
        return "<h2>Language and Communication Games</h2><p>Interactive storytelling and vocabulary games.</p>"
    elif activity_type == 'motor':
        return "<h2>Motor Skill Development</h2><p>Activities to improve coordination and movement.</p>"
    elif activity_type == 'creative':
        return "<h2>Creative Arts and Crafts</h2><p>Drawing, painting, and crafting activities.</p>"
    elif activity_type == 'math':
        return "<h2>Basic Math and Shapes</h2><p>Simple math exercises and shape recognition.</p>"
    else:
        return "<h2>Activity Not Found</h2>"

