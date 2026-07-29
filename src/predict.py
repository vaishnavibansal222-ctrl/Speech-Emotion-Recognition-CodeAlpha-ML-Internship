import os
import numpy as np
import librosa
import tensorflow as tf

# ==============================
# PATHS
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "speech_emotion_cnn.keras"
)

LABELS_PATH = os.path.join(
    BASE_DIR, "models", "emotion_labels.npy"
)

# ==============================
# LOAD MODEL AND LABELS
# ==============================

print("Loading speech emotion model...")

model = tf.keras.models.load_model(MODEL_PATH)
emotion_labels = np.load(LABELS_PATH, allow_pickle=True)

print("Model loaded successfully!")
print("Available emotions:", emotion_labels)


# ==============================
# FEATURE EXTRACTION
# ==============================

def extract_features(file_path):
    audio, sample_rate = librosa.load(
        file_path,
        duration=3,
        offset=0.5
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    # Make sure every audio file has exactly 174 time frames
    if mfcc.shape[1] < 174:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, 174 - mfcc.shape[1])),
            mode="constant"
        )
    else:
        mfcc = mfcc[:, :174]

    return mfcc


# ==============================
# PREDICTION
# ==============================

def predict_emotion(file_path):

    print("\nAnalyzing audio...")
    print("File:", file_path)

    features = extract_features(file_path)

    # CNN expects:
    # (samples, 40, 174, 1)

    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=-1)

    prediction = model.predict(features, verbose=0)

    predicted_index = np.argmax(prediction[0])
    confidence = prediction[0][predicted_index] * 100

    emotion = emotion_labels[predicted_index]

    print("\n==============================")
    print("   SPEECH EMOTION RESULT")
    print("==============================")
    print(f"Predicted Emotion : {emotion}")
    print(f"Confidence        : {confidence:.2f}%")
    print("==============================\n")


# ==============================
# MAIN PROGRAM
# ==============================

if __name__ == "__main__":

    audio_file = input(
        "Enter the path of your audio file: "
    ).strip('"')

    if not os.path.exists(audio_file):
        print("\nERROR: Audio file not found!")
    else:
        predict_emotion(audio_file)