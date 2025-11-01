**Author:** Kumar Abhishek  
**Date:** October 2025

A comprehensive deep learning and signal processing project for detecting R-peaks in ECG signals using multiple state-of-the-art methods on the MIT-BIH Arrhythmia Database.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Methods Implemented](#methods-implemented)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [References](#references)

---

## 🎯 Overview

This project implements **6 different approaches** for R-peak detection in ECG signals:

1. **Pan-Tompkins Algorithm** - Classical signal processing
2. **Wavelet Transform Detection** - Time-frequency analysis
3. **1D CNN (Convolutional Neural Network)** - Deep learning classification
4. **LSTM (Long Short-Term Memory)** - Recurrent neural network
5. **IncRes-UNet (RPNet)** - Advanced deep learning with Inception-Residual blocks
6. **Distance Transform Learning** - Novel approach using distance maps

The project provides a complete pipeline from data loading to model evaluation with comprehensive visualizations and metrics.

---

## ✨ Features

- **Multiple Detection Methods**: Compare classical and deep learning approaches
- **MIT-BIH Integration**: Direct loading from PhysioNet's MIT-BIH database
- **Comprehensive Evaluation**: Sensitivity, Precision, F1-Score, and confusion metrics
- **Professional Visualizations**: Publication-quality plots and comparisons
- **Modular Architecture**: Easy to extend with new methods
- **Training Pipeline**: Complete deep learning workflow with validation
- **Distance Transform Learning**: Novel approach for robust R-peak detection

---

## 🔬 Methods Implemented

### 1. Pan-Tompkins Algorithm
Classic ECG peak detection algorithm with:
- Bandpass filtering (5-15 Hz)
- Derivative-based QRS enhancement
- Squaring and moving window integration
- Adaptive thresholding with refractory period

### 2. Wavelet Transform Detection
Multi-scale analysis using:
- Stationary Wavelet Transform (SWT)
- Daubechies 4 (db4) wavelet
- 4-level decomposition
- Energy-based thresholding

### 3. 1D CNN Model
Deep convolutional architecture:
- 3 Conv1D layers (32, 64, 128 filters)
- Batch normalization and dropout
- MaxPooling for feature extraction
- Dense layers for classification
- Binary classification (peak/non-peak)

### 4. LSTM Model
Recurrent neural network:
- 3 LSTM layers (64, 32, 16 units)
- Return sequences for temporal modeling
- Dropout for regularization
- Temporal pattern recognition

### 5. IncRes-UNet (RPNet)
State-of-the-art architecture featuring:
- **Inception-Residual Blocks**: Multi-scale feature extraction (15, 17, 19, 21 kernel sizes)
- **U-Net Architecture**: 8 encoder and 8 decoder layers with skip connections
- **Distance Transform Prediction**: Novel approach for robust detection
- **SmoothL1 Loss**: Better handling of outliers
- **Deep Network**: 1024 channels at bottleneck for rich representations

### 6. Distance Transform Learning
Novel approach:
- Creates distance maps from R-peak locations
- Each point's distance to nearest R-peak
- Smooth continuous target (vs binary classification)
- Better gradient flow during training
- Post-processing extracts peaks from valleys

---

## 📁 Project Structure

```
R_Peak_Detection/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── MIT_BIH_R_Peak_Detection.ipynb    # Main Jupyter notebook (Methods 1-4)
├── advanced_rpeak_complete.py         # RPNet implementation (Methods 5-6)
├── MIT_BIH_arrhythmia_dataset.csv    # Dataset file
├── r_peak.pdf                        # Reference paper
└── advanced_rpeak_part2.txt          # Additional documentation
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)
- 8GB RAM minimum

### Step 1: Clone or Download
```bash
cd R_Peak_Detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```python
import numpy as np
import wfdb
import tensorflow as tf
import torch
print(\"All packages installed successfully!\")
```

---

## 💻 Usage

### Option 1: Jupyter Notebook (Methods 1-4)

Open the notebook for interactive exploration:
```bash
jupyter notebook MIT_BIH_R_Peak_Detection.ipynb
```

Run cells sequentially to:
1. Load MIT-BIH data from PhysioNet
2. Test Pan-Tompkins and Wavelet methods
3. Prepare training data for deep learning
4. Train CNN and LSTM models
5. Compare all methods with visualizations

### Option 2: Python Script (Advanced RPNet)

Run the advanced IncRes-UNet model:
```bash
python advanced_rpeak_complete.py
```

**Note**: Update the dataset path in `main()` function:
```python
filepath = 'MIT_BIH_arrhythmia_dataset.csv'  # Update this path
```

### Quick Start Example

```python
# Load a record
ecg_signal, true_peaks, fs = load_mitbih_record('100', sampto=10000)

# Detect peaks using Pan-Tompkins
pt_detector = PanTompkinsDetector(fs=fs)
detected_peaks = pt_detector.detect(ecg_signal)

# Calculate metrics
metrics = calculate_metrics(detected_peaks, true_peaks)
print(f\"Sensitivity: {metrics['Sensitivity']:.2f}%\")
print(f\"Precision: {metrics['Precision']:.2f}%\")
```

---

## 📊 Dataset

### MIT-BIH Arrhythmia Database

- **Source**: [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
- **Records**: 48 half-hour ECG recordings
- **Sampling Rate**: 360 Hz
- **Annotations**: Beat-by-beat labels by cardiologists
- **Classes**: Normal and various arrhythmia types

### Data Loading

The project loads data in two ways:

1. **Direct from PhysioNet** (Notebook):
```python
record = wfdb.rdrecord('100', pn_dir='mitdb')
annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')
```

2. **From CSV file** (Python script):
```python
df = pd.read_csv('MIT_BIH_arrhythmia_dataset.csv')
```

---

## 📈 Results

### Performance Metrics

| Method | Sensitivity | Precision | F1-Score |
|--------|-------------|-----------|----------|
| Pan-Tompkins | ~99.5% | ~99.3% | ~99.4% |
| Wavelet Transform | ~99.2% | ~99.0% | ~99.1% |
| 1D CNN | ~98.8% | ~98.5% | ~98.6% |
| LSTM | ~98.5% | ~98.3% | ~98.4% |
| IncRes-UNet (RPNet) | ~99.7% | ~99.6% | ~99.6% |

*Note: Results vary by record and configuration*

### Evaluation Criteria

- **Tolerance Window**: 75ms (27 samples at 360 Hz)
- **True Positive (TP)**: Detected peak within tolerance of true peak
- **False Positive (FP)**: Detected peak with no true peak nearby
- **False Negative (FN)**: Missed true peak
- **Sensitivity**: TP / (TP + FN) - Recall
- **Precision**: TP / (TP + FP) - Positive Predictive Value
- **F1-Score**: Harmonic mean of precision and sensitivity

---


### CNN Architecture

```
Input (300, 1)
├── Conv1D(32, kernel=5) + BatchNorm + ReLU + MaxPool + Dropout
├── Conv1D(64, kernel=5) + BatchNorm + ReLU + MaxPool + Dropout
├── Conv1D(128, kernel=3) + BatchNorm + ReLU + MaxPool + Dropout
├── Flatten
├── Dense(128) + ReLU + Dropout
├── Dense(64) + ReLU + Dropout
└── Dense(1) + Sigmoid
```

### LSTM Architecture

```
Input (300, 1)
├── LSTM(64, return_sequences=True) + Dropout
├── LSTM(32, return_sequences=True) + Dropout
├── LSTM(16) + Dropout
├── Dense(32) + ReLU + Dropout
└── Dense(1) + Sigmoid
```

### IncRes-UNet (RPNet) Architecture

**Inception-Residual Block**:
```
Input
├── Conv1D(1x1) dimension reduction
├── Branch1: Conv1D(15) ─┐
├── Branch2: Conv1D(17) ─┤
├── Branch3: Conv1D(19) ─┤─► Concatenate
├── Branch4: Conv1D(21) ─┘
├── Conv1D(1x1) combine
├── BatchNorm
└── Add (residual) + LeakyReLU
```

**Full U-Net**:
```
Encoder (8 layers):
Input → 64 → 128 → 256 → 512 → 1024 → 1024 → 1024 → 1024 (bottleneck)

Decoder (8 layers with skip connections):
1024 → 1024 → 1024 → 1024 → 512 → 256 → 128 → 64 → 1 (output)
```

### Distance Transform Concept

Traditional approach: Binary classification (peak/non-peak)
- Problem: Class imbalance, hard boundaries

Distance Transform approach: Regression on distance maps
- Advantage: Smooth continuous targets
- Each point predicts distance to nearest R-peak
- R-peaks are valleys (distance = 0)
- Better gradient flow during training

```python
# Creating Distance Transform
mask = np.zeros(len(ecg))
mask[r_peak_locations] = 1

# Distance of each point to nearest peak
distance_map = distance_transform_edt(1 - mask)
normalized_dt = distance_map / distance_map.max()
```

### Training Configuration

**CNN/LSTM**:
- Optimizer: Adam
- Loss: Binary Crossentropy
- Batch Size: 64
- Epochs: 20
- Train/Test Split: 80/20

**IncRes-UNet (RPNet)**:
- Optimizer: Adam (lr=0.05)
- Loss: SmoothL1
- Batch Size: 32
- Epochs: 500
- LR Schedule: Divide by 10 every 150 epochs
- Train/Val/Test Split: 86.4/9.6/4.0

---

## 📚 References

### Key Papers

1. **Pan-Tompkins Algorithm**
   - Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. IEEE Transactions on Biomedical Engineering, 32(3), 230-236.

2. **Wavelet-Based Detection**
   - Li, C., Zheng, C., & Tai, C. (1995). Detection of ECG characteristic points using wavelet transforms. IEEE Transactions on Biomedical Engineering, 42(1), 21-28.

3. **Deep Learning for R-Peak Detection**
   - Hong, S., et al. (2019). ENCASE: An ensemble classifier for ECG classification using expert features and deep neural networks. Computing in Cardiology, 46, 1-4.

4. **IncRes-UNet (RPNet)**
   - Referenced paper: \"A Deep Learning approach for robust R Peak detection in noisy ECG signals\"
   - Implements Distance Transform learning for improved robustness

5. **MIT-BIH Database**
   - Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH arrhythmia database. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50.

### Online Resources

- [PhysioNet MIT-BIH Database](https://physionet.org/content/mitdb/1.0.0/)
- [WFDB Python Package](https://wfdb.readthedocs.io/)
- [ECG Signal Processing Tutorial](https://www.robots.ox.ac.uk/~davidc/pubs/tt2015_ukras_tutorial.pdf)

---


## 📄 License

This project is for educational and research purposes. Please cite the original papers when using these methods in your research.

---

## 👨‍💻 Author

**Kumar Abhishek**  
October 2025


## 🎓 Acknowledgments

- **PhysioNet** for providing the MIT-BIH Arrhythmia Database
- **Pan & Tompkins** for the foundational QRS detection algorithm
- **WFDB Team** for excellent Python tools
- **Deep Learning Community** for open-source frameworks

---

---

**Last Updated**: November 2025  
**Version**: 1.0.0
