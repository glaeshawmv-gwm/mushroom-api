from flask import Flask, request, jsonify
import joblib
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
import os

# === Configuration ===
IMAGE_SIZE = (128, 128)
MODEL_PATH = 'random_forest_model.pkl'
PCA_PATH = 'pca.pkl'

# === Initialize Flask ===
app = Flask(__name__)

# === Load model and PCA ===
try:
    model = joblib.load(MODEL_PATH)
    pca = joblib.load(PCA_PATH)
    print("✅ Model and PCA loaded successfully.")
except Exception as e:
    print("❌ Error loading model or PCA:", e)
    model, pca = None, None


# === Feature Extraction (aligned with training) ===
def extract_features(image):
    """
    Extracts color histogram, HOG, LBP, color moments, and Laplacian variance.
    Applies PCA for dimensionality reduction.
    """
    image = cv2.resize(image, IMAGE_SIZE)

    # --- Color Histogram ---
    hist = cv2.calcHist([image], [0, 1, 2], None, [16, 16, 16],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # --- Convert to Grayscale ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- HOG Features ---
    hog_feat = hog(gray, orientations=12, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys',
                   feature_vector=True)

    # --- Local Binary Pattern ---
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # --- Color Moments ---
    color_moments = []
    for i in range(3):
        channel = image[:, :, i]
        color_moments.extend([
            np.mean(channel),
            np.std(channel),
            np.mean(np.abs(channel - np.mean(channel))**3) ** (1/3)
        ])

    # --- Laplacian Variance ---
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # --- Combine All Features ---
    combined = np.concatenate((hist, hog_feat, lbp_hist, color_moments, [lap_var])).reshape(1, -1)

    # --- Apply PCA ---
    return pca.transform(combined)


# === Routes ===
@app.route('/')
def home():
    return jsonify({'message': '✅ Mushroom Detection API (Random Forest + PCA) is running successfully!'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or pca is None:
        return jsonify({'error': 'Model or PCA not loaded properly.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded.'}), 400

    file = request.files['image']
    filename = file.filename.lower()

    # Validate file extension
    if not filename.endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({'error': 'Unsupported file type. Please upload a JPG or PNG image.'}), 400

    # Read and decode image
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Failed to decode the uploaded image.'}), 400

    try:
        # Extract features and predict
        features = extract_features(img)
        probs = model.predict_proba(features)[0]
        prediction_index = np.argmax(probs)
        prediction = model.classes_[prediction_index]
        confidence = float(probs[prediction_index])

        print(f"🧠 Prediction: {prediction} ({confidence:.2f})")

        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence, 4)
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({'error': str(e)}), 500


# === Run Flask app ===
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # ✅ Compatible with Render or local run
    app.run(host='0.0.0.0', port=port, debug=False)
