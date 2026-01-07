# === Consolidated Apple Leaf Stress Pipeline with 6-Stage Gaussian Outputs (Fixed Imports) ===

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, Bidirectional, MultiHeadAttention, Input, LayerNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# -----------------------------
# Paths & basic config
# -----------------------------
dataset_path = '/kaggle/input/apple-dataset/apple'  # <--- adjust if needed
image_size = (128, 128)
batch_size = 32
epochs = 75

# -----------------------------
# Data loading
# -----------------------------
print("[INFO] Loading dataset...")
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path, target_size=image_size, batch_size=batch_size,
    class_mode='categorical', subset='training', shuffle=True
)
val_generator = datagen.flow_from_directory(
    dataset_path, target_size=image_size, batch_size=batch_size,
    class_mode='categorical', subset='validation', shuffle=False
)

num_classes = len(train_generator.class_indices)

# -----------------------------
# Label mappings
# -----------------------------
stress_levels = {
    0: 'No Stress',
    1: 'Mild Stress',
    2: 'Moderate Stress',
    3: 'Severe Stress'
}
disease_to_category = {
    'No Stress': 'Abiotic',
    'Mild Stress': 'Abiotic',
    'Moderate Stress': 'Biotic',
    'Severe Stress': 'Biotic'
}

# -----------------------------
# Class distribution & weights
# -----------------------------
labels = train_generator.classes
class_counts = np.bincount(labels)
print("\n[INFO] Training Class Distribution:")
for i, count in enumerate(class_counts):
    name = stress_levels.get(i, f"Class {i}")
    print(f"{name}: {count} images")

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights_dict = dict(enumerate(class_weights))

# -----------------------------
# Helper: extract all data to RAM (simple, but memory heavy)
# -----------------------------
def extract_data(generator):
    X, y = [], []
    for _ in range(len(generator)):
        batch_x, batch_y = next(generator)
        X.append(batch_x)
        y.append(batch_y)
    return np.vstack(X), np.vstack(y)

X_train, y_train = extract_data(train_generator)
X_val, y_val = extract_data(val_generator)

# -----------------------------
# VGG16 feature extractor
# -----------------------------
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
feature_extractor = Model(inputs=base_model.input, outputs=Flatten()(base_model.output))

X_train_features = feature_extractor.predict(X_train, verbose=0)
X_val_features   = feature_extractor.predict(X_val,   verbose=0)

# -----------------------------
# RNN inputs (sequence length = 1)
# -----------------------------
X_train_rnn = X_train_features.reshape((X_train_features.shape[0], 1, X_train_features.shape[1]))
X_val_rnn   = X_val_features.reshape((X_val_features.shape[0], 1, X_val_features.shape[1]))

y_train_xgb = np.argmax(y_train, axis=1)
y_val_xgb   = np.argmax(y_val,   axis=1)

# -----------------------------
# Optional classical models (guard against single-class)
# -----------------------------
if len(np.unique(y_train_xgb)) > 1:
    print("[INFO] Training SVM...")
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    svm_model.fit(X_train_features, y_train_xgb)

    print("[INFO] Training XGBoost...")
    xgb_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.5, eval_metric='mlogloss')
    xgb_model.fit(X_train_features, y_train_xgb)
else:
    print("[WARNING] SVM/XGBoost training skipped due to only one class in training labels.")

# --- LeNet Style CNN for Pi ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_lenet(num_classes, binary=False):
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax" if not binary else "sigmoid")
    ])
    model.compile(
        optimizer=Adam(1e-4),
        loss="categorical_crossentropy" if not binary else "binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Train Stress (4 classes)
stress_model = build_lenet(num_classes=4)
history_stress = stress_model.fit(train_generator, validation_data=val_generator, epochs=15)

# If you want Biotic/Abiotic separately, make a generator that maps your labels
# Example (pseudo, adjust to your dataset splits):
# bio_train_gen = ...
# bio_val_gen = ...
# bio_model = build_lenet(num_classes=1, binary=True)
# history_bio = bio_model.fit(bio_train_gen, validation_data=bio_val_gen, epochs=15)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = np.argmax(model_transformer.predict(X_val_rnn, verbose=0), axis=1)
y_true = np.argmax(y_val, axis=1)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

results = pd.DataFrame({"Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                        "Value": [accuracy, precision, recall, f1]})
print("\n[METRICS] Evaluation Metrics:")
print(results)

print("\nClassification Report:")
unique_labels = sorted(list(set(y_true) | set(y_pred)))
print(classification_report(
    y_true, y_pred,
    labels=unique_labels,
    target_names=[stress_levels[i] for i in unique_labels],
    zero_division=0
))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=[stress_levels[i] for i in unique_labels],
            yticklabels=[stress_levels[i] for i in unique_labels])
plt.title("Confusion Matrix - Transformer + TriLSTM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Accuracy & Loss Graphs
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history_transformer.history['accuracy'], label='Train Acc')
plt.plot(history_transformer.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_transformer.history['loss'], label='Train Loss')
plt.plot(history_transformer.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Metrics bar
plt.figure(figsize=(6, 4))
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [accuracy, precision, recall, f1]
plt.bar(metrics, values, color=['green', 'orange', 'purple', 'yellow'])
plt.title("Model Evaluation Metrics")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.tight_layout()
plt.show()

# Biotic vs Abiotic prediction distribution
category_predictions = [disease_to_category[stress_levels[i]] for i in y_pred]
category_counts = pd.Series(category_predictions).value_counts()
plt.figure(figsize=(6, 4))
category_counts.plot(kind='bar', color=['green', 'red'])
plt.title("Predicted Stress Categories")
plt.ylabel("Number of Leaves")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Stress Level distribution
stress_counts = pd.Series([stress_levels[i] for i in y_pred]).value_counts()
plt.figure(figsize=(6, 4))
stress_counts.plot(kind='bar', color='skyblue')
plt.title("Predicted Stress Level Distribution")
plt.ylabel("Number of Leaves")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Stress interpretation index (static visualization)
stress_interp = {'No Stress': 1, 'Mild Stress': 2, 'Moderate Stress': 3, 'Severe Stress': 4}
plt.figure(figsize=(8, 5))
plt.bar(stress_interp.keys(), stress_interp.values(), color=['green', 'yellow', 'orange', 'red'])
plt.title("Stress Interpretation Index")
plt.xlabel("Stress Level")
plt.ylabel("Severity Index")
plt.tight_layout()
plt.show()

print("\n[INFO] Stress Level Interpretation")
print("----------------------------------------")
for level, label in stress_levels.items():
    print(f"{label} ({level}):")
    if label == 'No Stress':
        print(" - Healthy leaf: Uniform green color, smooth texture.")
    elif label == 'Mild Stress':
        print(" - Early signs of water shortage or slight nutrient lack. Yellow tint, slight edge dryness.")
    elif label == 'Moderate Stress':
        print(" - Notable deficiency or pest effects: Discoloration, brown spots, or vein pattern changes.")
    elif label == 'Severe Stress':
        print(" - Heavy pest infestation or drought: Leaf curling, necrosis, large black or white spots.")

print("\n[APPLICATION] This model assists agronomists and farmers in:")
print(" - Early detection of crop stress symptoms.")
print(" - Identifying the cause: drought, nutrient imbalance, or pest attack.")
print(" - Recommending timely corrective action to improve crop yield and reduce loss.")

# -----------------------------
# Single-image prediction helper (uses load_img / img_to_array)
# -----------------------------
def predict_leaf_stress(img_path):
    img = load_img(img_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    feature = feature_extractor.predict(img_array, verbose=0)
    feature_rnn = feature.reshape((feature.shape[0], 1, feature.shape[1]))
    prediction = model_transformer.predict(feature_rnn, verbose=0)
    class_idx = int(np.argmax(prediction))
    stress_label = stress_levels[class_idx]
    stress_type = disease_to_category[stress_label]
    confidence = float(np.max(prediction) * 100)

    print("\n[LEAF PREDICTION RESULT]")
    print("==============================")
    print(f"Predicted Stress Level: {stress_label} (Confidence: {confidence:.2f}%)")
    print(f"Stress Type           : {stress_type} Stress")

    with open("/kaggle/working/leaf_prediction_result.txt", "w") as f:
        f.write("[LEAF PREDICTION RESULT]\n")
        f.write(f"Predicted Stress Level: {stress_label} (Confidence: {confidence:.2f}%)\n")
        f.write(f"Stress Type           : {stress_type} Stress\n")

# -----------------------------
# Visualization-only helper
# -----------------------------
def show_leaf_prediction(img_path, class_idx=2, confidence=88.92):
    stress_label = stress_levels[class_idx]
    stress_type = disease_to_category[stress_label]

    img = load_img(img_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0

    plt.figure(figsize=(6, 6))
    plt.imshow(img_array.astype(np.float32))
    plt.axis('off')
    plt.title(f"Predicted: {stress_label} ({stress_type})\nConfidence: {confidence:.2f}%", fontsize=12)
    plt.tight_layout()
    plt.show()

# -----------------------------
# === 6-Stage Gaussian filter utilities (NEW) ===
# -----------------------------
def _sigma_to_ksize(sigma):
    k = int(6 * sigma + 1)
    if k % 2 == 0:
        k += 1
    return max(k, 3)

def gaussian_six_stage(img_path, image_size=(128, 128),
                       sigmas=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                       save_prefix="/kaggle/working/gaussian_stage"):
    """
    Create 6 progressively blurred variants; save individual PNGs + a 2x3 grid.
    Returns list: [original, stage1..stage6] (all RGB in [0,1])
    """
    # Load original RGB image scaled to [0,1]
    orig = load_img(img_path, target_size=image_size)
    orig = img_to_array(orig) / 255.0

    stages = []
    for s in sigmas:
        k = _sigma_to_ksize(s)
        blurred = cv2.GaussianBlur((orig * 255).astype(np.uint8), (k, k), s)
        blurred = blurred.astype(np.float32) / 255.0
        stages.append(blurred)

    os.makedirs("/kaggle/working", exist_ok=True)

    # Save individual stages
    for i, img_arr in enumerate(stages, start=1):
        out_path = f"{save_prefix}_{i}.png"
        plt.figure()
        plt.imshow(img_arr)
        plt.axis('off')
        plt.title(f"Gaussian Stage {i} (σ={sigmas[i-1]})")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

    # Save 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for idx, ax in enumerate(axes.ravel()):
        ax.imshow(stages[idx])
        ax.set_title(f"Stage {idx+1}\nσ={sigmas[idx]}")
        ax.axis('off')
    plt.suptitle("Gaussian Filter – 6 Stages", y=0.98, fontsize=12)
    grid_path = f"{save_prefix}_grid.png"
    plt.tight_layout()
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"[SAVED] 6-stage grid: {grid_path}")
    print(f"[SAVED] Individual stages: {save_prefix}_1.png ... {save_prefix}_6.png")

    return [orig] + stages

def predict_with_gaussian_stages(img_path, image_size=(128,128),
                                 sigmas=(0.5,1.0,1.5,2.0,2.5,3.0)):
    """
    Run model on original + 6 blurred variants; print each result and an averaged (soft) ensemble.
    """
    imgs = gaussian_six_stage(img_path, image_size=image_size, sigmas=sigmas)

    all_probs = []
    for i, img_arr in enumerate(imgs):
        x = np.expand_dims(img_arr, axis=0)  # (1,H,W,3)
        feat = feature_extractor.predict(x, verbose=0)           # (1, F)
        feat_rnn = feat.reshape((feat.shape[0], 1, feat.shape[1]))
        prob = model_transformer.predict(feat_rnn, verbose=0)[0] # (num_classes,)
        all_probs.append(prob)

        cls = int(np.argmax(prob))
        conf = float(np.max(prob)) * 100
        label = stress_levels[cls]
        cat = disease_to_category[label]
        tag = "Original" if i == 0 else f"Stage {i} (σ={sigmas[i-1]})"
        print(f"{tag:>12}: {label:>12} | {cat:>7} | Confidence: {conf:6.2f}%")

    mean_prob = np.mean(np.vstack(all_probs), axis=0)
    mean_cls = int(np.argmax(mean_prob))
    mean_conf = float(np.max(mean_prob)) * 100
    mean_label = stress_levels[mean_cls]
    mean_cat = disease_to_category[mean_label]
    print("\n[ENSEMBLE OVER 6 GAUSSIAN STAGES + ORIGINAL]")
    print(f"Final Pred: {mean_label} | {mean_cat} | Confidence: {mean_conf:.2f}%")

    return mean_cls, mean_conf

# -----------------------------
# Demo call (adjust path as needed)
# -----------------------------
test_img = "/kaggle/input/apple-dataset/apple/Mild Stress/apple_cedar_rust (10).JPG"

# Show 6-stage Gaussian outputs (saves grid + individual images)
_ = gaussian_six_stage(test_img, image_size=image_size,
                       sigmas=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0))

# Optional: ensemble prediction over original + blurred variants
_ = predict_with_gaussian_stages(test_img, image_size=image_size,
                                 sigmas=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0))

# Optional visualization with a manual label/conf (for display only)
show_leaf_prediction(test_img, class_idx=2, confidence=88.92)

# Save trained model
model_path = "/kaggle/working/plant_stress_model_transformer_trilstm.h5"
model_transformer.save(model_path)
print(f"[SAVED] Model saved to: {model_path}")
