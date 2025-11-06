import os
import cv2
import numpy as np
import joblib
import random
import time
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, ParameterGrid
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler

# ========== CONFIGURATION ==========
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATASET_DIR = 'mushroom_dataset'
NON_MUSHROOM_DIR = 'non_mushroom_images'
IMAGE_SIZE = (128, 128)
MODEL_PATH = 'random_forest_model.pkl'
PCA_PATH = 'pca.pkl'
USE_PCA = True
PCA_VARIANCE = 0.99  # retain 99% variance

# ========== FEATURE EXTRACTION ==========
def extract_features(image):
    image = cv2.resize(image, IMAGE_SIZE)

    # Color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [16, 16, 16],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # HOG features
    hog_feat = hog(gray, orientations=12, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys',
                   feature_vector=True)

    # Local Binary Pattern
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # Color moments
    color_moments = []
    for i in range(3):
        channel = image[:, :, i]
        color_moments.extend([
            np.mean(channel),
            np.std(channel),
            np.mean(np.abs(channel - np.mean(channel))**3) ** (1/3)
        ])

    # Laplacian variance
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return np.concatenate((hist, hog_feat, lbp_hist, color_moments, [lap_var]))


# ========== AUGMENTATION ==========
def augment_image(image):
    augmented = [cv2.flip(image, 1), cv2.flip(image, 0)]
    for angle in [-25, -10, 10, 25]:
        M = cv2.getRotationMatrix2D((IMAGE_SIZE[0]//2, IMAGE_SIZE[1]//2), angle, 1)
        augmented.append(cv2.warpAffine(image, M, IMAGE_SIZE))
    augmented.append(cv2.convertScaleAbs(image, alpha=1.3, beta=35))
    augmented.append(cv2.convertScaleAbs(image, alpha=0.7, beta=-35))
    return augmented


# ========== LOAD DATA ==========
def load_dataset():
    features, labels = [], []
    valid_exts = ('.png', '.jpg', '.jpeg')
    valid_count, skipped_count = 0, 0

    for label in os.listdir(DATASET_DIR):
        class_dir = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(valid_exts):
                skipped_count += 1
                continue
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                skipped_count += 1
                continue

            valid_count += 1
            features.append(extract_features(img))
            labels.append(label)

            for aug_img in augment_image(img):
                features.append(extract_features(aug_img))
                labels.append(label)

    # Non-mushroom images
    if os.path.exists(NON_MUSHROOM_DIR):
        for img_name in os.listdir(NON_MUSHROOM_DIR):
            if not img_name.lower().endswith(valid_exts):
                continue
            img_path = os.path.join(NON_MUSHROOM_DIR, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                valid_count += 1
                features.append(extract_features(img))
                labels.append("not_mushroom")

    print(f"📂 Loaded {valid_count} valid images, skipped {skipped_count} invalid/non-image files.")
    return np.array(features), np.array(labels)


# ========== MAIN ==========
def main():
    print("🔍 Loading dataset...")
    X, y = load_dataset()
    print(f"✅ Total samples: {len(X)}, Classes: {set(y)}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # PCA
    if USE_PCA:
        print(f"📉 Applying PCA (retain {int(PCA_VARIANCE*100)}% variance)...")
        pca = PCA(n_components=PCA_VARIANCE, svd_solver='full', random_state=SEED)
        X = pca.fit_transform(X)
        joblib.dump(pca, PCA_PATH)
        print(f"✅ PCA reduced to {X.shape[1]} components")

    # ========== Manual Grid Search with Live Progress ==========
    print("🔎 Running GridSearchCV manually (with live progress)...")

    param_grid = {
        'n_estimators': [400, 600, 800],
        'max_depth': [40, 60, 80],
        'min_samples_split': [2, 3, 5],
        'max_features': ['sqrt', 'log2']
    }

    grid = list(ParameterGrid(param_grid))
    best_score = 0
    best_params = None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    start_time = time.time()
    progress = tqdm(total=len(grid), desc="🚀 Training Progress", ncols=100)

    for i, params in enumerate(grid, start=1):
        model = RandomForestClassifier(**params, random_state=SEED, class_weight='balanced', n_jobs=-1)
        scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
        mean_score = scores.mean()

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

        progress.set_postfix({
            "Current": f"{mean_score:.4f}",
            "Best": f"{best_score:.4f}"
        })
        progress.update(1)

    progress.close()
    elapsed = time.time() - start_time

    print(f"\n🏆 Best Params: {best_params}")
    print(f"✅ Best Cross-Validation Score: {best_score:.4f}")
    print(f"⏱️ Total Training Time: {elapsed/60:.2f} minutes")

    # Train final model
    print("\n🧠 Training final model with best parameters...")
    best_model = RandomForestClassifier(**best_params, random_state=SEED, class_weight='balanced', n_jobs=-1)
    best_model.fit(X, y)

    # Evaluate
    print("\n✅ Evaluating Final Model...")
    y_pred = best_model.predict(X)
    print("\n📋 Classification Report:")
    print(classification_report(y, y_pred))

    print("\n📈 Summary:")
    print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y, y_pred, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(y, y_pred, average='weighted'):.4f}")

    joblib.dump(best_model, MODEL_PATH)
    print(f"\n💾 Saved optimized model to '{MODEL_PATH}'")


if __name__ == "__main__":
    main()
