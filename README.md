# 🕵️ DeepFake Image Detection Using Deep Learning

An AI-powered image classification system that predicts whether a face image is **REAL or FAKE** using deep learning, transfer learning, and EfficientNetB0. The project includes a Gradio web application that displays the predicted class along with the probabilities for both classes.

## 📌 Project Overview

AI-generated and digitally manipulated face images are becoming increasingly realistic. This project explores how deep learning can help distinguish genuine face images from fake ones.

The system uses a pre-trained **EfficientNetB0** convolutional neural network, adapted through transfer learning and fine-tuning. Users can upload an image through a simple web interface and receive a REAL or FAKE prediction.

**This project is an educational prototype, not a definitive authenticity verification tool.**

## ✨ Key Features

- 🖼️ **Image Classification:** Classifies uploaded face images as REAL or FAKE.
- 🧠 **Transfer Learning:** Uses the pre-trained EfficientNetB0 architecture with ImageNet weights.
- 🔧 **Fine-Tuning:** Fine-tunes the last 30 backbone layers to improve classification performance.
- 🔄 **Data Augmentation:** Applies horizontal flipping, rotation, zoom, and contrast adjustments during training.
- 📊 **Model Evaluation:** Evaluates accuracy, precision, recall, F1-score, and confusion matrices.
- 🎚️ **Threshold Optimization:** Selects a decision threshold using validation data.
- 🌐 **Interactive Web Application:** Built with Gradio for easy image uploading and prediction.
- 📈 **Probability Output:** Displays REAL and FAKE probabilities as percentages.

## 🛠️ Technologies Used

- **Programming Language:** Python 3.12
- **Deep Learning Framework:** TensorFlow / Keras
- **Model Architecture:** EfficientNetB0
- **Web Interface:** Gradio
- **Numerical Computing:** NumPy
- **Evaluation:** Scikit-learn
- **Development Environment:** Jupyter Notebook and VS Code
- **Hardware Used for Training:** NVIDIA GeForce RTX 3050 Laptop GPU

## 🧠 Model Architecture

The model uses EfficientNetB0 as its feature extraction backbone and adds a custom classification head.

**Architecture workflow:**

1. Input image resized to 224 × 224 pixels in RGB format.
2. Data augmentation during training.
3. Pre-trained EfficientNetB0 feature extraction.
4. Global average pooling.
5. Dropout layer with a rate of 0.3.
6. Dense layer with 128 neurons and ReLU activation.
7. Dropout layer with a rate of 0.2.
8. Final sigmoid output for binary classification.

The model was trained in two phases:

- **Phase 1 – Transfer Learning:** The EfficientNetB0 backbone was frozen while the classification head was trained for five epochs.
- **Phase 2 – Fine-Tuning:** The last 30 backbone layers were unfrozen and fine-tuned for another five epochs using a lower learning rate.

The final model contains approximately 4.21 million parameters.

## 📂 Dataset

The dataset contains **190,335 face images** divided into training, validation, and test sets.

| Dataset Split | Fake Images | Real Images | Total |
|---|---:|---:|---:|
| Training | 70,001 | 70,001 | 140,002 |
| Validation | 19,641 | 19,787 | 39,428 |
| Testing | 5,492 | 5,413 | 10,905 |
| **Total** | **95,134** | **95,201** | **190,335** |

All images were resized to 224 × 224 pixels. The supplied dataset partitions were retained for training, validation, and testing.

*Note: The original dataset source and licence were not recorded in the project documentation.*

## 📊 Model Performance

The model was evaluated on 10,905 held-out test images.

The decision threshold of 0.20 was selected using validation data and then applied to the test set.

| Evaluation Metric | Threshold 0.50 | Threshold 0.20 |
|---|---:|---:|
| Test Accuracy | 79.20% | **83.36%** |
| Real Precision | 92.63% | 87.59% |
| Real Recall | 63.13% | 77.44% |
| Real F1-Score | 75.08% | 82.20% |
| Fake Recall | 95.05% | 89.18% |

Fine-tuning improved validation accuracy from **77.57% to 90.55%**.

The validation-selected threshold of 0.20 improved test accuracy to **83.36%** and provided a more balanced result between the REAL and FAKE classes.

These results apply only to the dataset used in this experiment. Performance on independent datasets has not been established.

## 🌐 Web Application

The project includes a Gradio application implemented in `app.py`.

### How it works

1. Upload a face image.
2. Click **ANALYZE IMAGE**.
3. The application preprocesses the image and passes it to the trained model.
4. The model estimates the probabilities of REAL and FAKE.
5. The application displays the predicted label, a brief analysis message, and both class probabilities.

The application loads the saved model `deepfake_detector_final.keras` and the decision threshold from `threshold.txt`.

## 📁 Project Structure

```text
Deep-Fake-Image-Detection/
│
├── DeepFake_Detection.ipynb
│   └── Model training, evaluation and prediction
│
├── app.py
│   └── Gradio web application
│
├── models/
│   ├── best_frozen_model.keras
│   ├── best_finetuned_model.keras
│   ├── deepfake_detector_final.keras
│   └── threshold.txt
│
└── README.md
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.12
- pip
- Git
- The trained model files and threshold file

### 1. Clone the repository

```bash
git clone https://github.com/unforeseenallen/Deep-Fake-Image-Detection.git
cd Deep-Fake-Image-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

Install the packages used by the application:

```bash
pip install tensorflow gradio numpy
```

Install any additional packages required by the notebook separately.

### 4. Verify the model files

Make sure these files exist in the paths configured in `app.py`:

- `models/deepfake_detector_final.keras`
- `models/threshold.txt`

If necessary, update `MODEL_PATH` and `THRESHOLD_PATH` in `app.py` to match your local folder structure.

### 5. Run the application

```bash
python app.py
```

Open the local URL displayed in the terminal, normally `http://127.0.0.1:7860`.

## ⚠️ Limitations

- The model has not been evaluated on independent datasets.
- A noticeable performance gap exists between validation and test results.
- The model may perform differently on images from unseen generators or images affected by compression, resizing, or filters.
- The application assumes that the uploaded image contains a face; face detection and cropping are not implemented.
- The displayed probabilities are model scores and have not been calibrated.
- Incorrect predictions are possible. The system should not be used as the sole basis for decisions about image authenticity.

## 🚀 Future Improvements

- Evaluate performance on independent datasets and compressed images.
- Investigate the validation-to-test performance gap.
- Experiment with additional model architectures and fine-tuning strategies.
- Add ROC-AUC, precision-recall curves, and probability calibration.
- Integrate face detection and cropping.
- Add visual explanations using Grad-CAM or similar techniques.
- Improve configuration and deployment portability.

## 🎓 Project Information

**Project Title:** DeepFake Image Detection Using Deep Learning

**Author:** Allen K Abi

**Department:** Department of Computer Science

**Institution:** Yenepoya Institute of Arts, Science, Commerce & Management, Yenepoya (Deemed to be University), Mangaluru, Karnataka, India.

## 📜 Disclaimer

This project was developed for educational and experimental purposes. Its predictions are not proof that an image is authentic or manipulated. Human review and independent verification are recommended before making consequential decisions.

## ⭐ Acknowledgements

This project uses the EfficientNetB0 architecture, TensorFlow/Keras, and Gradio. The implementation builds on established work in deep learning, transfer learning, and synthetic-image detection.

---

**If you find this project useful, consider giving the repository a star!** ⭐
