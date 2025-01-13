import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the CSV data into a pandas DataFrame
heart_data = pd.read_csv('hb.csv')

# Check for missing values
if heart_data.isnull().sum().any():
    print("Missing values detected. Please clean the data.")
else:
    print("No missing values detected.")

# Check the distribution of the target variable
print("Target variable distribution:\n", heart_data['target'].value_counts())

# Split the data into features and target variable
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the logistic regression model with training data
model.fit(X_train, Y_train)

# Save the model to a file
joblib.dump(model, 'heart_disease_model.pkl')
print("Model saved as 'heart_disease_model.pkl'.")

# Load the model from the file (if needed)
loaded_model = joblib.load('heart_disease_model.pkl')
print("Model loaded successfully.")

# Accuracy on training data
X_train_prediction = loaded_model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy on training data:', training_data_accuracy)

# Accuracy on test data
X_test_prediction = loaded_model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy on test data:', test_data_accuracy)

# Function to make predictions based on user input
def make_prediction(input_data):
    # Convert input data to a numpy array and reshape it
    input_data_as_numpy_array = np.array(input_data).reshape(1, -1)
    
    # Make the prediction
    prediction = loaded_model.predict(input_data_as_numpy_array)
    
    # Return the result based on the prediction
    return "The person does not have heart disease." if prediction[0] == 0 else "The person has heart disease."

# User input for prediction
try:
    age = int(input("Enter age: "))
    sex = int(input("Enter sex (0 = Female, 1 = Male): "))
    cp = int(input("Enter chest pain type (0-3): "))
    trestbps = int(input("Enter resting blood pressure (in mm Hg): "))
    chol = int(input("Enter cholesterol (in mg/dl): "))
    fbs = int(input("Enter fasting blood sugar > 120 mg/dl (0 = No, 1 = Yes): "))
    restecg = int(input("Enter resting electrocardiographic results (0-2): "))
    thalach = int(input("Enter max heart rate achieved: "))
    exang = int(input("Enter exercise induced angina (0 = No, 1 = Yes): "))
    oldpeak = float(input("Enter oldpeak: "))

    # Prepare the input data for prediction
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak)
    
    # Make and print the prediction
    result = make_prediction(input_data)
    print(result)

except ValueError as e:
    print(f"Invalid input: {e}. Please enter numeric values.")
import pickle
with open('heart_disease_model.pkl', 'wb') as file:
    pickle.dump(model, file)
with open('heart_disease_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
