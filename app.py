from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- Add this line
import joblib
import cv2
import numpy as np
from skimage.feature import hog

# === Configuration ===
IMAGE_SIZE = (100, 100)
MODEL_PATH = 'random_forest_model.pkl'
PCA_PATH = 'pca.pkl'

# === Load model and PCA ===
model = joblib.load(MODEL_PATH)
pca = joblib.load(PCA_PATH)

app = Flask(__name__)
CORS(app)  # <-- Enable CORS for cross-origin requests from Flutter

# === Health Check Route ===
@app.route('/')
def index():
    return 'API is up and running!', 200  # <-- Helpful for testing

def extract_features(image):
    image = cv2.resize(image, IMAGE_SIZE)

    # Color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # HOG features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)

    # Combine and reshape
    combined = np.concatenate((hist, hog_feat)).reshape(1, -1)
    return pca.transform(combined)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image_bytes = file.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Could not decode image'}), 400

    try:
        features = extract_features(img)
        probs = model.predict_proba(features)[0]
        prediction_index = np.argmax(probs)
        prediction = model.classes_[prediction_index]
        confidence = float(probs[prediction_index])

        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence, 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
