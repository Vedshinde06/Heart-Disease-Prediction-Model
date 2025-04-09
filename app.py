import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import traceback
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Set up model path
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'XGBoost_model.pkl')

# Define feature names based on your specification
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Define expected data types for each feature
FEATURE_TYPES = {
    'age': int,
    'sex': int,
    'cp': int,
    'trestbps': int,
    'chol': int,
    'fbs': int,
    'restecg': int,
    'thalach': int,
    'exang': int,
    'oldpeak': float,
    'slope': int,
    'ca': int,
    'thal': int
}

# Class names for binary classification
CLASS_NAMES = ['0', '1']  # Negative (0) and Positive (1)

# Load the model
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded successfully from {MODEL_PATH}")
    MODEL_LOADED = True
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    MODEL_LOADED = False

def validate_input(data):
    """Validate the input data against expected features and types."""
    errors = []
    
    # Check if all features are present
    for feature in FEATURE_NAMES:
        if feature not in data:
            errors.append(f"Missing feature: {feature}")
    
    # Check for extra features
    for feature in data:
        if feature not in FEATURE_NAMES:
            errors.append(f"Unexpected feature: {feature}")
    
    # Check data types
    for feature, value in data.items():
        if feature in FEATURE_TYPES:
            expected_type = FEATURE_TYPES[feature]
            # Allow integer values for float fields, but not vice versa
            if expected_type == float:
                if not (isinstance(value, (int, float))):
                    errors.append(f"Feature '{feature}' should be a number, got {type(value).__name__}")
            elif not isinstance(value, expected_type):
                errors.append(f"Feature '{feature}' should be {expected_type.__name__}, got {type(value).__name__}")
    
    return errors

def preprocess_input(data):
    """Convert input data to the format expected by the model."""
    # If data is a list, convert to dict using feature names
    if isinstance(data, list):
        if len(data) != len(FEATURE_NAMES):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} features, got {len(data)}")
        data = dict(zip(FEATURE_NAMES, data))
    
    # Validate input data
    errors = validate_input(data)
    if errors:
        raise ValueError(f"Input validation failed: {', '.join(errors)}")
    
    # Create DataFrame with features in the correct order
    df = pd.DataFrame([data], columns=FEATURE_NAMES)
    
    # Convert data types
    for feature, dtype in FEATURE_TYPES.items():
        if dtype == int:
            df[feature] = df[feature].astype(int)
        elif dtype == float:
            df[feature] = df[feature].astype(float)
    
    return df

@app.route('/')
def home():
    """Home page with frontend interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions."""
    if not MODEL_LOADED:
        return jsonify({
            "status": "error",
            "message": "Model is not loaded"
        }), 503
    
    try:
        # Get data from request
        data = request.get_json(force=True)
        
        if 'features' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing 'features' key in request"
            }), 400
            
        features = data['features']
        
        # Preprocess the input data
        try:
            input_df = preprocess_input(features)
        except ValueError as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 400
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Get probability score (if applicable)
        try:
            probability = model.predict_proba(input_df)[0][1]  # Probability of class 1
        except:
            probability = None
        
        # Prepare response
        result = {
            "status": "success",
            "prediction": int(prediction),
            "prediction_label": "Heart Disease" if prediction == 1 else "No Heart Disease"
        }
        
        if probability is not None:
            result["probability"] = float(probability)
        
        return jsonify(result)
        
    except Exception as e:
        # Log the error for debugging
        print(f"Error making prediction: {e}")
        traceback.print_exc()
        
        return jsonify({
            "status": "error",
            "message": f"Error processing request: {str(e)}"
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED,
        "model_path": MODEL_PATH
    })

@app.route('/sample', methods=['GET'])
def sample():
    """Return a sample input for testing."""
    sample_input = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }
    
    return jsonify({
        "status": "success",
        "sample_input": sample_input,
        "usage": "Send a POST request to /predict with the 'features' object"
    })

if __name__ == '__main__':
    # Get port from environment variable or use 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=True)