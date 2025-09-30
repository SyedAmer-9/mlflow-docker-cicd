import pandas as pd
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model directly from the .pkl file
# This is much simpler and more reliable inside Docker.
model_filename = 'model.pkl'
loaded_model = joblib.load(model_filename)

@app.route('/predict', methods=["POST"])
def predict():
    json_data = request.get_json()
    df = pd.DataFrame(json_data['data'])
    prediction = loaded_model.predict(df).tolist()
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)