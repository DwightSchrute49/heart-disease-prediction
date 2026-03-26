# Heart Disease Prediction (Machine Learning)

## Overview

This project predicts the likelihood of heart disease using clinical health parameters.

---

## Model

- Algorithm: Random Forest Classifier
- Hyperparameters:
  - n_estimators = 200
  - max_depth = 5

---

## Performance

- Training Accuracy: 93.98%
- Testing Accuracy: 87.03%

The small difference between training and testing accuracy indicates good generalization with limited overfitting.

---

## 🚀 How to Run

1️⃣ Install dependencies:
pip install -r requirements.txt

2️⃣ Train the model:
python train.py

3️⃣ Run prediction:
python predict.py
