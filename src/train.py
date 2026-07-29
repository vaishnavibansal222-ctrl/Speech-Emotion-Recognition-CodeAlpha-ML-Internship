"""
Task 2: Emotion Recognition from Speech
Dataset: RAVDESS
Features: MFCC
Model: CNN
"""

import os
import warnings
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "RAVDESS")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "speech_emotion_cnn.keras")

# ============================================================
# 2. EMOTION MAPPING
# ============================================================

# RAVDESS emotion codes:
# 01 = neutral
# 02 = calm
# 03 = happy
# 04 = sad
# 05 = angry
# 06 = fearful
# 07 = disgust
# 08 = surprised

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# ============================================================
# 3. MFCC FEATURE EXTRACTION
# ============================================================

def extract_mfcc(file_path, n_mfcc=40, max_pad_len=174):
    """
    Load an audio file and extract MFCC features.

    Parameters:
        file_path: Path to .wav file
        n_mfcc: Number of MFCC coefficients
        max_pad_len: Fixed time dimension

    Returns:
        MFCC feature matrix
    """

    try:
        audio, sample_rate = librosa.load(
            file_path,
            sr=22050,
            duration=3,
            offset=0.5
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=n_mfcc
        )

        # Normalize MFCC values
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

        # Make every sample the same size
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(
                mfcc,
                pad_width=((0, 0), (0, pad_width)),
                mode="constant"
            )
        else:
            mfcc = mfcc[:, :max_pad_len]

        return mfcc

    except Exception as error:
        print(f"Error processing {file_path}: {error}")
        return None


# ============================================================
# 4. LOAD RAVDESS DATASET
# ============================================================

def load_dataset():

    features = []
    labels = []

    print("\nLoading RAVDESS dataset...")
    print("-" * 60)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset folder not found:\n{DATASET_PATH}"
        )

    for actor_folder in sorted(os.listdir(DATASET_PATH)):

        actor_path = os.path.join(DATASET_PATH, actor_folder)

        if not os.path.isdir(actor_path):
            continue

        if not actor_folder.startswith("Actor_"):
            continue

        for filename in os.listdir(actor_path):

            if not filename.lower().endswith(".wav"):
                continue

            # Example:
            # 03-01-05-01-02-01-12.wav
            parts = filename.split("-")

            if len(parts) < 3:
                continue

            emotion_code = parts[2]

            if emotion_code not in EMOTION_MAP:
                continue

            emotion = EMOTION_MAP[emotion_code]

            file_path = os.path.join(actor_path, filename)

            mfcc = extract_mfcc(file_path)

            if mfcc is not None:
                features.append(mfcc)
                labels.append(emotion)

    print(f"Total audio files processed: {len(features)}")

    return np.array(features), np.array(labels)


# ============================================================
# 5. LOAD DATA
# ============================================================

X, y = load_dataset()

if len(X) == 0:
    raise RuntimeError(
        "No audio files were found. "
        "Check that Actor_01 ... Actor_24 are inside dataset/RAVDESS."
    )

print("\nFeature shape:", X.shape)
print("Labels:", len(y))

# ============================================================
# 6. ENCODE LABELS
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

class_names = label_encoder.classes_
num_classes = len(class_names)

y_categorical = to_categorical(
    y_encoded,
    num_classes=num_classes
)

print("\nEmotion classes:")
for index, emotion in enumerate(class_names):
    print(f"{index}: {emotion}")

# ============================================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y_categorical,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# Need encoded labels for stratification of second split
y_temp_labels = np.argmax(y_temp, axis=1)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp_labels
)

print("\nDataset split:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# ============================================================
# 8. RESHAPE FOR CNN
# ============================================================

# CNN expects:
# samples, height, width, channels

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

input_shape = X_train.shape[1:]

print("\nCNN input shape:", input_shape)

# ============================================================
# 9. BUILD CNN MODEL
# ============================================================

model = Sequential([
    
    Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        input_shape=input_shape
    ),

    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(
        128,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.30),

    Flatten(),

    Dense(256, activation="relu"),
    Dropout(0.40),

    Dense(num_classes, activation="softmax")
])

# ============================================================
# 10. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN Model Summary:")
model.summary()

# ============================================================
# 11. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\nStarting CNN training...")
print("=" * 60)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=[
        early_stopping,
        model_checkpoint
    ],
    verbose=1
)

# ============================================================
# 13. TEST MODEL
# ============================================================

print("\nEvaluating model on test data...")
print("-" * 60)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ============================================================
# 14. PREDICTIONS
# ============================================================

y_probability = model.predict(X_test, verbose=0)

y_pred = np.argmax(y_probability, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)

print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")

# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print("\nClassification Report:")
print("=" * 60)
print(report)

# Save classification report
report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report)

# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Speech Emotion Recognition - Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.tight_layout()

confusion_matrix_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(confusion_matrix_path, dpi=300)
plt.close()

# ============================================================
# 17. TRAINING ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

accuracy_graph_path = os.path.join(
    OUTPUT_DIR,
    "accuracy_graph.png"
)

plt.savefig(accuracy_graph_path, dpi=300)
plt.close()

# ============================================================
# 18. TRAINING LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

loss_graph_path = os.path.join(
    OUTPUT_DIR,
    "loss_graph.png"
)

plt.savefig(loss_graph_path, dpi=300)
plt.close()

# ============================================================
# 19. SAVE FINAL MODEL
# ============================================================

model.save(MODEL_PATH)

# Save class labels
labels_path = os.path.join(
    MODEL_DIR,
    "emotion_labels.npy"
)

np.save(labels_path, class_names)

# ============================================================
# 20. FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(f"Test Accuracy       : {test_accuracy * 100:.2f}%")
print(f"Model saved         : {MODEL_PATH}")
print(f"Labels saved        : {labels_path}")
print(f"Confusion matrix    : {confusion_matrix_path}")
print(f"Accuracy graph      : {accuracy_graph_path}")
print(f"Loss graph          : {loss_graph_path}")
print(f"Classification report: {report_path}")

print("\nDetected emotions:")
for emotion in class_names:
    print(f" - {emotion}")

print("\nTask 2: Emotion Recognition from Speech is complete.")