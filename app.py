from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import requests
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for browser requests

# Load model
model = load_model("deepfake_model.h5")

def preprocess_image(image_url):
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()  # Raise error for bad requests
        img = Image.open(BytesIO(response.content))
        img = img.resize((224, 224))  # Resize for model input
        img = np.array(img) / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        print("Error processing image:", e)
        return None

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    image_url = data.get('image_url', '')

    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    img = preprocess_image(image_url)
    if img is None:
        return jsonify({"error": "Could not process image"}), 400

    # Predict
    prediction = model.predict(img)
    confidence = float(prediction[0][0])  # Convert to float
    is_deepfake = confidence > 0.5  # Adjust threshold as needed

    return jsonify({
        "image_url": image_url,
        "is_deepfake": is_deepfake,
        "confidence": confidence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
