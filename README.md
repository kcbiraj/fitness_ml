# 🏋️‍♂️ Activity Recognition and Repetition Detection — Machine Learning

This project focuses on recognizing human activities and counting exercise repetitions using accelerometer and gyroscope data. Machine learning models and signal processing techniques are applied to classify activities and detect exercise repetitions accurately.

---

## 🎯 Project Objective

- Classify human activities (e.g., squats, deadlifts).
- Detect exercise repetitions using peak detection.
- Apply feature engineering, outlier removal, and machine learning models for improved performance.

---

## 📊 Dataset Overview

The dataset includes:

- **Accelerometer and Gyroscope Data** — Measurements of linear acceleration and rotational velocity.
- **Activity Label** — Activity classes (e.g., squat, deadlift).
- **Time** — Timestamp of each data point.

The data is collected from wearable sensors during exercises.

---

## 🧪 Methodology

- **Data Preprocessing**: Cleaning and noise removal.
- **Feature Engineering**: Extracting features like mean, standard deviation, and peaks.
- **Model Training**: Using classifiers like Random Forest, KNN, and SVM.
- **Repetition Detection**: Peak detection for counting repetitions.

### 🧠 Machine Learning Models:

- **Random Forest Classifier**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Neural Networks** (for activity classification)

### 📊 Signal Processing Techniques:

- **Low-Pass Filtering**: Removes high-frequency noise.
- **Peak Detection**: Detects exercise repetition start and end.

---

## 🧮 Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

## 📈 Model Performance

| Model                        | Accuracy   |
|------------------------------|------------|
| Random Forest Classifier      | 98.875%      |
| K-Nearest Neighbors (KNN)     | 99.762%      |
| Support Vector Machine (SVM)  | 96.612%      |

---

## 📌 Interpretation

- **Random Forest** achieved the best performance at 92.5% accuracy.
- **KNN** and **SVM** also performed well but slightly less accurately.

---

## ✅ Conclusion

The project successfully classified activities and counted repetitions. The **Random Forest Classifier** was the most effective model for activity recognition, and the peak detection method provided reliable repetition counts.

---

## 📚 References

- **Project Data**: This dataset is from [Dave Ebbelaar’s Master's Thesis](https://docs.datalumina.io/xLAtq6PNUsMcfG).

- **Code Reference**: This repository is based on the book "[Machine Learning for the Quantified Self](https://ml4qs.org)" by Mark Hoogendoorn and Burkhardt Funk, published by Springer in 2018. The code is available on [GitHub](https://github.com/kcbiraj/ML4QS).

- **License**: The code is made available under the GNU public license. Please cite the book in your publications:  
  Hoogendoorn, M., & Funk, B. (2018). *Machine Learning for the Quantified Self – On the Art of Learning from Sensory Data*. Springer.
