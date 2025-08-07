import os
import cv2
import numpy as np
import joblib
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from skimage.feature import hog

# Config
DATASET_DIR = 'mushroom_dataset'
IMAGE_SIZE = (100, 100)
MODEL_PATH = 'random_forest_model.pkl'
PCA_PATH = 'pca.pkl'
AUGMENTATIONS_PER_IMAGE = 8
USE_PCA = True
PCA_COMPONENTS = 150  # Tuned for ~95% variance

def extract_features(image):
    image = cv2.resize(image, IMAGE_SIZE)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)

    return np.concatenate((hist, hog_feat))

def augment_image(image):
    augmented = []

    # Horizontal & vertical flips
    augmented.append(cv2.flip(image, 1))  # Horizontal
    augmented.append(cv2.flip(image, 0))  # Vertical

    # Rotation
    angle = random.randint(-20, 20)
    M = cv2.getRotationMatrix2D((IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2), angle, 1.0)
    augmented.append(cv2.warpAffine(image, M, IMAGE_SIZE))

    # Brightness & contrast
    bright = cv2.convertScaleAbs(image, alpha=1.1, beta=25)
    dark = cv2.convertScaleAbs(image, alpha=0.9, beta=-25)
    augmented.extend([bright, dark])

    # Blur & noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    noise = image.copy()
    cv2.randn(noise, (0), (20))
    noisy = cv2.add(image, noise)
    augmented.extend([blurred, noisy])

    random.shuffle(augmented)
    return augmented[:AUGMENTATIONS_PER_IMAGE]

def load_dataset():
    features, labels = [], []
    for label in os.listdir(DATASET_DIR):
        class_dir = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(class_dir): continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None: continue

            features.append(extract_features(img))
            labels.append(label)

            for aug_img in augment_image(img):
                features.append(extract_features(aug_img))
                labels.append(label)

    return np.array(features), np.array(labels)

def main():
    print("🔍 Loading dataset...")
    X, y = load_dataset()
    print(f"✅ Total samples: {len(X)}")

    # PCA
    if USE_PCA:
        print("📉 Applying PCA for dimensionality reduction...")
        pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
        X = pca.fit_transform(X)
        joblib.dump(pca, PCA_PATH)
        print(f"✅ PCA reduced to {X.shape[1]} components")
    else:
        pca = None

    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in stratified_split.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    print("🌲 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("✅ Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\n📈 Summary Metrics:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")

    print(f"\n💾 Saving model to '{MODEL_PATH}'...")
    joblib.dump(model, MODEL_PATH)
    print("✅ Done!")

if __name__ == '__main__':
    main()
