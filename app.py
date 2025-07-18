from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import urllib.request

model_path = "model.pkl"
model = None

if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?export=download&id=1miUawFu13A3UrNtLd3lP5IwPDSkGkENM"
    urllib.request.urlretrieve(url, model_path)

app = Flask(__name__)

@app.route("/")
def home():
    return "ML Ops API is running."

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        model = joblib.load(model_path)
    data = request.get_json(force=True)
    input_data = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(input_data)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
