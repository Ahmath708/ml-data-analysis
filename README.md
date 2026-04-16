# ML Data Analysis

A comprehensive machine learning project demonstrating supervised and unsupervised learning techniques using scikit-learn and TensorFlow.

## Features

- **Data Preprocessing**: Feature engineering, StandardScaler normalization
- **Supervised Learning**:
  - Linear Regression & Ridge & Lasso
  - Decision Trees
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - Neural Networks (TensorFlow/Keras)
- **Unsupervised Learning**:
  - K-Means Clustering
- **Hyperparameter Tuning**:
  - GridSearchCV
  - Cross-validation
- **Model Evaluation**:
  - R² Score, RMSE, MAE
  - Feature Importance

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Results

| Model | R² | RMSE |
|-------|-----|------|
| Random Forest (Tuned) | 0.9950 | 35,546 |
| Random Forest | 0.9949 | 36,151 |
| Decision Tree | 0.9836 | 64,573 |
| Ridge/Lasso | 0.7150 | ~269,500 |
| Neural Network | 0.6077 | 316,195 |

## Tech Stack

- Python
- scikit-learn
- TensorFlow
- Pandas
- NumPy
- Matplotlib

## License

MIT