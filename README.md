# Predicting Recidivism using Deep Learning

Predict recidivism using a deep learning model trained on 48,000+ records from NIJ, COMPAS, and North Carolina. This project implements end-to-end data processing and domain-specific feature engineering in TensorFlow/Keras, achieving a stable ~69% test accuracy.

## Features
* **Custom Feature Engineering:** Transforms raw criminal records into behavioral, socio-economic, and severity indicators[cite: 1].
* ** Neural Network:** Multi-layer Perceptron (MLP) built with TensorFlow/Keras using He weight initialization and dropout regularization[cite: 1].
* **Early Stopping & Optimization:** Employs Adam optimization with early stopping callbacks to prevent overfitting during training[cite: 1].
* **Balanced Evaluation:** Comprehensive performance assessment using precision, recall, and F1-score across balanced classes[cite: 1].

## Dataset & Engineered Features
The model operates on a merged dataset containing demographic, criminal history, and social stability metrics[cite: 1].

### Primary Raw & Encoded Features
* `Age`, `Gender_Encoded`, `Education_Level_Encoded`, `Offense_Type_Encoded`[cite: 1]
* `Prior_Arrests_Count`, `Prior_Felony_Arrests`, `Prior_Misdemeanor_Arrests`, `Prior_Violent_Arrests`[cite: 1]
* `Sentence_Years`, `Employed`, `Substance_Abuse`[cite: 1]

### Engineered Features
```python
Arrest_Rate             = Prior_Arrests_Count / (Age + 1)
Felony_Ratio            = Prior_Felony_Arrests / (Prior_Arrests_Count + 1e-5)
Is_Young_Offender       = 1 if Age < 25 else 0
Criminal_Severity_Score = (Prior_Violent_Arrests * 3) + (Prior_Felony_Arrests * 2) + Prior_Misdemeanor_Arrests
Instability_Index       = (1 - Employed) + Substance_Abuse
Sentence_per_Arrest     = Sentence_Years / (Prior_Arrests_Count + 1)
Felony_Density          = Prior_Felony_Arrests / (Age + 1)
```[cite: 1]

## Model Architecture
The network is structured as follows:
* **Preprocessing:** `StandardScaler` applied to all feature columns[cite: 1].
* **Layer 1:** Dense (64 units, ReLU activation, HeNormal initialization) + Dropout (0.2)[cite: 1]
* **Layer 2:** Dense (32 units, ReLU activation, HeNormal initialization) + Dropout (0.2)[cite: 1]
* **Output:** Dense (1 unit, Sigmoid activation)[cite: 1]
* **Loss & Optimizer:** Binary Crossentropy, Adam (`lr=0.0001`)[cite: 1]

## Performance
Evaluated on a holdout test set of 9,630 records (80/20 train-test split)[cite: 1]:

| Metric | Non-Recidivism (0) | Recidivism (1) | Overall / Avg |
| :--- | :--- | :--- | :--- |
| **Precision** | 0.70[cite: 1] | 0.67[cite: 1] | 0.68[cite: 1] |
| **Recall** | 0.70[cite: 1] | 0.66[cite: 1] | 0.68[cite: 1] |
| **F1-Score** | 0.70[cite: 1] | 0.67[cite: 1] | 0.68[cite: 1] |
| **Accuracy** | — | — | **68% – 69%**[cite: 1] |


