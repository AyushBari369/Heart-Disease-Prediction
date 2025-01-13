from flask import Flask, render_template, request
import numpy as np
import joblib

# Load the trained logistic regression model
model = joblib.load('heart_disease_model.pkl')

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    data = [
        int(request.form['age']),
        int(request.form['sex']),
        int(request.form['cp']),
        int(request.form['trestbps']),
        int(request.form['chol']),
        int(request.form['fbs']),
        int(request.form['restecg']),
        int(request.form['thalach']),
        int(request.form['exang']),
        float(request.form['oldpeak']),
        #int(request.form['slope']),
        #int(request.form['ca']),
        #int(request.form['thal']),
    ]

    # Convert input data to a numpy array
    input_data_as_numpy_array = np.array(data).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data_as_numpy_array)

    # Return result based on prediction
    result = "The person does not have heart disease." if prediction[0] == 0 else "The person has heart disease."
    
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)

