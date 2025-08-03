# Gesture Recognition using CNN2D + Poisson Encoder + BiLSTM

> A deep learning pipeline for recognizing hand gestures using FFT-transformed data, a 2D CNN for spatial feature extraction, spike-based encoding with a Poisson rate encoder, and a bidirectional LSTM classifier.

---

## 📊 Overview

This project builds an end-to-end gesture classification model using frequency domain features. It leverages:

- 📈 **FFT-transformed 4D tensor data** from multiple gesture trials
- 🧠 **2D CNN** for spatial feature extraction
- ⚡ **Poisson Rate Encoder** to simulate spike-based neural input
- 🔁 **BiLSTM** for sequence modeling
- ✅ **Accuracy & Confusion Matrix visualization**

---

## 📁 Dataset

- Input: 4D numpy tensor of shape `(num_gestures, num_trials, 16, 513)`
- FFT transformation was applied to the original time-series sensor data.
- Gestures used:
  - `Pointer`, `Power`, `Open`, `Tripod`, `Lateral`, `Rest`

---

## 🧱 Model Architecture

```text
Input (1, 16, 513)
   │
[ CNN2D Block ]
   ↓
(128, 4, 4) → Flattened to 2048-dim
   ↓
[ PoissonRateEncoder (T=10) ]
   ↓
(10, 2048) → spike-like inputs
   ↓
[ BiLSTM ]
   ↓
[ Linear → Softmax ]
   ↓
Gesture Prediction (6 classes)
