from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
import re

app = Flask(__name__)
model = load_model("gesture_model.h5")

# Load label list
with open("gesture_labels.txt", "r") as f:
    labels = [line.strip() for line in f]

def preprocess_image(data_url):
    # Decode base64 image
    img_str = re.search(r'base64,(.*)', data_url).group(1)
    img_bytes = base64.b64decode(img_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Resize and normalize
    img = cv2.resize(img, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_data = data['image']
    img = preprocess_image(image_data)

    preds = model.predict(img)[0]
    pred_index = np.argmax(preds)
    confidence = float(preds[pred_index])
    label = labels[pred_index]

    return jsonify({"label": label, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)
