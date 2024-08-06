import numpy as np
from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.joblib')

# Load the vectorizer
vectorizer = joblib.load('vectorizer.joblib')

# Home route
@app.route('/')
def home():
    return render_template('open.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input message from the form
        data = request.form['message']
        data = data.lower()  # Convert to lowercase
        
        # Transform the input text using the vectorizer
        data_vectorized = vectorizer.transform([data])
        
        # Predict using the logistic regression model
        prediction = model.predict(data_vectorized)[0]
        
        print("Received form data:", data)
        print("Prediction:", prediction)
        
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        print("Error during prediction:", e)  # Debugging statement
        return render_template('result.html', prediction=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
