from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
import re
import os

app = Flask(__name__)

# Load model
MODEL_PATH = "gesture_model.h5"
LABELS_PATH = "gesture_labels.txt"

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    raise FileNotFoundError("Model or labels file is missing.")

model = load_model(MODEL_PATH)

# Load label list
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]


def preprocess_image(data_url):
    try:
        # Extract base64 string
        match = re.search(r"base64,(.*)", data_url)
        if not match:
            raise ValueError("Invalid image data")
        img_str = match.group(1)

        # Decode to bytes and then to NumPy array
        img_bytes = base64.b64decode(img_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Image decoding failed")

        # Resize and normalize
        img = cv2.resize(img, (64, 64))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)  # (64, 64, 1)
        img = np.expand_dims(img, axis=0)   # (1, 64, 64, 1)
        return img
    except Exception as e:
        raise ValueError(f"Error in image preprocessing: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_data = data.get('image')
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        img = preprocess_image(image_data)
        preds = model.predict(img)[0]
        pred_index = int(np.argmax(preds))
        confidence = float(preds[pred_index])
        label = labels[pred_index]

        return jsonify({"label": label, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload():
    return predict()  # same functionality as predict


if __name__ == "__main__":
    app.run(debug=True)
