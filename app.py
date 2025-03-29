from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np

# Load the trained XGBoost model (JSON format)
model = xgb.Booster()
model.load_model("updated_model.json")  # Ensure this is the correct path

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ XGBoost Model API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.json  
        features = np.array(data["features"]).reshape(1, -1)  # Convert to NumPy array

        # Convert to DMatrix (required for Booster)
        dmatrix = xgb.DMatrix(features)

        # Make prediction
        prediction = model.predict(dmatrix)[0]

        return jsonify({"prediction": float(prediction)})  # Return as JSON
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the API
if __name__ == '__main__':
    app.run(debug=True)
