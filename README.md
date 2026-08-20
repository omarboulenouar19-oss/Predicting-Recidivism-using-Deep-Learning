# Predicting Recidivism using Deep Learning

Predict recidivism using a deep learning model trained on 48,000+ records from NIJ, COMPAS, and North Carolina. This project implements end-to-end data processing and domain-specific feature engineering in TensorFlow/Keras, achieving a stable ~69% test accuracy.

## Features
* **Custom Feature Engineering:** Transforms raw criminal records into behavioral, socio-economic, and severity indicators.
* **Neural Network:** Multi-layer Perceptron (MLP) built with TensorFlow/Keras using He weight initialization and dropout regularization.
* **Early Stopping & Optimization:** Employs Adam optimization with early stopping callbacks to prevent overfitting during training.
* **Balanced Evaluation:** Comprehensive performance assessment using precision, recall, and F1-score across balanced classes.

## Dataset & Engineered Features
The model operates on a merged dataset containing demographic, criminal history, and social stability metrics.

### Primary Raw & Encoded Features
* `Age`, `Gender_Encoded`, `Education_Level_Encoded`, `Offense_Type_Encoded`
* `Prior_Arrests_Count`, `Prior_Felony_Arrests`, `Prior_Misdemeanor_Arrests`, `Prior_Violent_Arrests`
* `Sentence_Years`, `Employed`, `Substance_Abuse`

### Engineered Features
```python
Arrest_Rate             = Prior_Arrests_Count / (Age + 1)
Felony_Ratio            = Prior_Felony_Arrests / (Prior_Arrests_Count + 1e-5)
Is_Young_Offender       = 1 if Age < 25 else 0
Criminal_Severity_Score = (Prior_Violent_Arrests * 3) + (Prior_Felony_Arrests * 2) + Prior_Misdemeanor_Arrests
Instability_Index       = (1 - Employed) + Substance_Abuse
Sentence_per_Arrest     = Sentence_Years / (Prior_Arrests_Count + 1)
Felony_Density          = Prior_Felony_Arrests / (Age + 1)


## Model Architecture
The network is structured as follows:
Preprocessing: `StandardScaler` applied to all feature columns.
Layer 1: Dense (64 units, ReLU activation, HeNormal initialization) + Dropout (0.2)
Layer 2: Dense (32 units, ReLU activation, HeNormal initialization) + Dropout (0.2)
Output: Dense (1 unit, Sigmoid activation)
Loss & Optimizer: Binary Crossentropy, Adam (`lr=0.0001`)




