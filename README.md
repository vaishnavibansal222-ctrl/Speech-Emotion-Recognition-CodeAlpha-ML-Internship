<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=250&color=0:0F172A,50:7C3AED,100:A78BFA&text=Speech%20Emotion%20Recognition&fontAlignY=38&fontSize=42&fontColor=FFFFFF&animation=fadeIn&desc=CNN%20%7C%20Deep%20Learning%20%7C%20Audio%20Classification&descAlignY=58"/>
</p>

<h1 align="center">🎙️ Speech Emotion Recognition using CNN</h1>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Poppins&weight=700&size=25&duration=3000&pause=1000&color=A78BFA&center=true&vCenter=true&width=900&lines=Speech+Emotion+Recognition;Built+using+Python+and+TensorFlow;CNN+%7C+Deep+Learning+%7C+Audio+Processing;RAVDESS+Dataset;Machine+Learning+Internship+Project" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/Librosa-Audio%20Processing-8A2BE2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CodeAlpha-ML%20Internship-2563EB?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Project-Speech%20Emotion%20Recognition-7C3AED?style=for-the-badge"/>
  <img src="https://img.shields.io/github/license/vaishnavibansal222-ctrl/Speech-Emotion-Recognition-CodeAlpha-ML-Internship?style=for-the-badge"/>
</p>

A Machine Learning project that recognizes human emotions from speech audio using a Convolutional Neural Network (CNN).

## 📌 Project Overview

Speech contains important emotional information such as tone, pitch, and intensity. This project uses audio feature extraction and deep learning to classify speech into different emotional categories.

The model is trained on the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

## 🔄 Project Workflow

```mermaid
flowchart LR
    A["RAVDESS Dataset"] --> B["Audio Preprocessing"]
    B --> C["MFCC Feature Extraction"]
    C --> D["CNN Model"]
    D --> E["Emotion Classification"]

    E --> F["Model Evaluation"]

    F --> F1["Accuracy"]
    F --> F2["Confusion Matrix"]
    F --> F3["Classification Report"]
    F --> F4["Accuracy & Loss Graphs"]

    D --> G["speech_emotion_cnn.keras"]
    E --> H["predict.py"]
```
## 🧠 Machine Learning Pipeline

```text
🎙️ Speech Audio
       ↓
🎵 Audio Preprocessing
       ↓
📊 Feature Extraction
       ↓
🔊 MFCC Features
       ↓
🧠 CNN Model
       ↓
🏋️ Model Training
       ↓
😊 Emotion Prediction
       ↓
📈 Model Evaluation
```

## 🎯 Emotions Recognized

The model recognizes 8 emotions:

- 😠 Angry
- 😌 Calm
- 🤢 Disgust
- 😨 Fearful
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprised

## 📊 Dataset

The project uses the RAVDESS dataset.

Dataset statistics used in this project:

- Total audio files processed: **1,440**
- Training samples: **1,152**
- Validation samples: **144**
- Testing samples: **144**
- Number of emotion classes: **8**

## 🧠 Model

A Convolutional Neural Network (CNN) is used for speech emotion classification.

### Input Features

Audio files are processed using **Librosa** to extract relevant audio features.

Feature shape:
(40, 174)

CNN input shape:

(40, 174, 1)

## 📈 Model Performance

Final test accuracy:

46.53%

The project also generates:

* Accuracy graph
* Loss graph
* Confusion matrix
* Classification report
  
 ## 🗂️ Project Structure
```mermaid
flowchart TD
    A["Speech-Emotion-Recognition"]

    A --> B["dataset"]
    B --> B1["RAVDESS"]
    B1 --> B2["Actor_01 ... Actor_24"]

    A --> C["models"]
    C --> C1["emotion_labels.npy"]
    C --> C2["speech_emotion_cnn.keras"]

    A --> D["outputs"]
    D --> D1["accuracy_graph.png"]
    D --> D2["loss_graph.png"]
    D --> D3["confusion_matrix.png"]
    D --> D4["classification_report.txt"]

    A --> E["src"]
    E --> E1["train.py"]
    E --> E2["predict.py"]

    A --> F[".gitignore"]
    A --> G["requirements.txt"]
    A --> H["README.md"]
```
## ⚙️ Technologies Used
* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Librosa
* Scikit-learn
* Matplotlib
* Seaborn
## 🚀 How to Run
1. Clone the repository
git clone https://github.com/vaishnavibansal222-ctrl/Speech-Emotion-Recognition-CodeAlpha-ML-Internship.git

2. Open the project
cd Speech-Emotion-Recognition-CodeAlpha-ML-Internship

6. Create a virtual environment
python -m venv venv

7. Activate the environment
Windows PowerShell:
venv\Scripts\Activate.ps1

8. Install dependencies
pip install -r requirements.txt

9. Predict emotion from an audio file
python src/predict.py

## 🔍 Example Prediction
SPEECH EMOTION RESULT

Predicted Emotion : fearful
Confidence        : 100.00%

## 📁 Generated Files

| File | Description |
|---|---|
| `speech_emotion_cnn.keras` | Trained CNN model used for speech emotion classification |
| `emotion_labels.npy` | Stores the mapping of model output classes to emotion labels |
| `accuracy_graph.png` | Shows training and validation accuracy across epochs |
| `loss_graph.png` | Shows training and validation loss across epochs |
| `confusion_matrix.png` | Visualizes actual vs. predicted emotion classifications |
| `classification_report.txt` | Contains precision, recall, F1-score, and support for each emotion |

## 📈 Results

The trained CNN model can classify speech audio into multiple emotional categories. The generated graphs and classification report are available in the outputs/ folder.

⚠️ Note

The model is intended as an educational machine learning project. The reported accuracy depends on the dataset, feature extraction method, model architecture, and training configuration.

## 👩‍💻 Author

Vaishnavi Bansal

Machine Learning / AI Enthusiast

⭐ Acknowledgement
* RAVDESS Dataset
* CodeAlpha Internship
  
# 🤝 Let's Connect

<p align="center">

<a href="YOUR_LINKEDIN_URL">
<img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white"/>
</a>

<a href="mailto:YOUR_EMAIL">
<img src="https://img.shields.io/badge/Email-Contact-EA4335?style=for-the-badge&logo=gmail&logoColor=white"/>
</a>

<a href="https://github.com/vaishnavibansal222-ctrl">
<img src="https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white"/>
</a>

</p>

## 💼 CodeAlpha Internship
**Machine Learning Intern — CodeAlpha**
This project was developed as part of my Machine Learning Internship at CodeAlpha.


## ⭐ If you find this project useful, consider giving it a star!
