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
SCALER_PATH = 'scaler.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

# === Initialize Flask ===
app = Flask(__name__)

# === Load model, PCA, scaler, label encoder ===
try:
    model = joblib.load(MODEL_PATH)
    pca = joblib.load(PCA_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    print("✅ Model, PCA, scaler, and label encoder loaded successfully.")
except Exception as e:
    print("❌ Error loading model or preprocessing objects:", e)
    model, pca, scaler, le = None, None, None, None

# === Feature Extraction ===
def extract_features(image):
    image = cv2.resize(image, IMAGE_SIZE)

    # Color histogram
    hist = cv2.calcHist([image], [0,1,2], None, [16,16,16],[0,256,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # HOG
    hog_feat = hog(gray, orientations=12, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), block_norm='L2-Hys', feature_vector=True)

    # LBP
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0,10), range=(0,9))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # Color moments
    color_moments = []
    for i in range(3):
        ch = image[:,:,i]
        color_moments.extend([np.mean(ch), np.std(ch), np.mean(np.abs(ch-np.mean(ch))**3)**(1/3)])

    # Laplacian
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    combined = np.concatenate((hist, hog_feat, lbp_hist, color_moments, [lap_var])).reshape(1,-1)
    return combined

# === Routes ===
@app.route('/')
def home():
    return jsonify({'message': '✅ Mushroom Detection API running successfully!'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if None in (model, pca, scaler, le):
        return jsonify({'error': 'Model or preprocessing objects not loaded properly.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    file = request.files['image']
    filename = file.filename.lower()
    if not filename.endswith(('.jpg','.jpeg','.png')):
        return jsonify({'error': 'Unsupported file type.'}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Failed to decode image.'}), 400

    try:
        features = extract_features(img)
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)

        pred_encoded = model.predict(features_pca)
        prediction = le.inverse_transform(pred_encoded)[0]
        confidence = float(np.max(model.predict_proba(features_pca)))

        print(f"🧠 Prediction: {prediction} ({confidence:.2f})")
        return jsonify({'prediction': prediction, 'confidence': round(confidence,4)})

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({'error': str(e)}), 500

# === Run Flask app ===
if __name__ == '__main__':
    port = int(os.environ.get("PORT",5000))
    app.run(host='0.0.0.0', port=port, debug=False)
