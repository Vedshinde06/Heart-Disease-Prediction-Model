import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import traceback

app = Flask(__name__)

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
    """Home page with API documentation."""
    model_status = "Loaded and ready" if MODEL_LOADED else "Not loaded"
    feature_list = ", ".join([f"{name} ({FEATURE_TYPES[name].__name__})" for name in FEATURE_NAMES])
    
    return f"""
    <html>
        <head>
            <title>Heart Disease Prediction API</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #333; }}
                code {{ background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
                pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                .status {{ padding: 10px; border-radius: 5px; display: inline-block; }}
                .loaded {{ background-color: #d4edda; color: #155724; }}
                .not-loaded {{ background-color: #f8d7da; color: #721c24; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Heart Disease Prediction API</h1>
            <p>Model Status: <span class="status {'loaded' if MODEL_LOADED else 'not-loaded'}">{model_status}</span></p>
            
            <h2>Model Information:</h2>
            <ul>
                <li><strong>Model Type:</strong> XGBoost Binary Classification</li>
                <li><strong>Classification:</strong> Binary (0: No Heart Disease, 1: Heart Disease)</li>
            </ul>
            
            <h2>Required Features:</h2>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Type</th>
                    <th>Description</th>
                </tr>
                <tr><td>age</td><td>int</td><td>Age in years</td></tr>
                <tr><td>sex</td><td>int</td><td>Sex (1 = male; 0 = female)</td></tr>
                <tr><td>cp</td><td>int</td><td>Chest pain type (0-3)</td></tr>
                <tr><td>trestbps</td><td>int</td><td>Resting blood pressure (in mm Hg)</td></tr>
                <tr><td>chol</td><td>int</td><td>Serum cholesterol in mg/dl</td></tr>
                <tr><td>fbs</td><td>int</td><td>Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)</td></tr>
                <tr><td>restecg</td><td>int</td><td>Resting electrocardiographic results (0-2)</td></tr>
                <tr><td>thalach</td><td>int</td><td>Maximum heart rate achieved</td></tr>
                <tr><td>exang</td><td>int</td><td>Exercise induced angina (1 = yes; 0 = no)</td></tr>
                <tr><td>oldpeak</td><td>float</td><td>ST depression induced by exercise relative to rest</td></tr>
                <tr><td>slope</td><td>int</td><td>Slope of the peak exercise ST segment (0-2)</td></tr>
                <tr><td>ca</td><td>int</td><td>Number of major vessels (0-3) colored by fluoroscopy</td></tr>
                <tr><td>thal</td><td>int</td><td>Thalassemia (0-3)</td></tr>
            </table>
            
            <h2>API Usage:</h2>
            <h3>Make a prediction</h3>
            <p><strong>Endpoint:</strong> <code>POST /predict</code></p>
            <p><strong>Request Format (JSON):</strong></p>
            <p>Option 1: Feature object</p>
            <pre>{{
    "features": {{
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
    }}
}}</pre>
            <p>Option 2: Feature array (in exact order listed above)</p>
            <pre>{{
    "features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
}}</pre>
            <p><strong>Response Format:</strong></p>
            <pre>{{
    "status": "success",
    "prediction": 0 or 1,
    "prediction_label": "No Heart Disease" or "Heart Disease",
    "probability": probability value between 0 and 1
}}</pre>
        </body>
    </html>
    """

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