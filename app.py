from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import urllib.request

if not os.path.exists("model.pkl"):
    url = "https://drive.google.com/uc?export=download&id=1RiNpIGiIpA-QbL7tpwmYASyfyBZX8E_B"
    urllib.request.urlretrieve(url, "model.pkl")

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "ML Ops API is running."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(input_data)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
