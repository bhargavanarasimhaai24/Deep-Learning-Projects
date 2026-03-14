# Multi-Layer Perceptron (MLP) for Binary Classification

Implementation of a Multi-Layer Perceptron neural network using TensorFlow/Keras for binary classification tasks. Demonstrates data preprocessing, 
model building, training, evaluation, and visualization on two datasets: a custom Ovarian cancer dataset (349 samples, 50 features) and the 
Breast Cancer Wisconsin dataset (569 samples, 32 features). 

## Features
- Data loading and exploration with Pandas.
- Normalization (MinMaxScaler, StandardScaler).
- Train-test split and model training with Adam optimizer, binary cross-entropy loss.
- Evaluation: Confusion matrix, accuracy score (84% on ovarian data, 95% on breast cancer).
- Loss plotting with Matplotlib.
- Advanced: Early stopping, validation split.

## Datasets
| Dataset | Samples | Features | Target | Accuracy |
|---------|---------|----------|--------|----------|
| Blood Tests (ovariantotal.csv) | 349 | 49 | Binary (TYPE: 0/1) | 84% |
| Breast Cancer Wisconsin | 569 | 31 | Binary (M/B) | 95-99% |

## Quick Start
1. Install dependencies: `pip install tensorflow scikit-learn pandas numpy matplotlib`.
2. Run `Multi_Layer_Perceptron.ipynb` in Jupyter/Colab.
3. Datasets: Download `ovariantotal.csv` (custom) and breast-cancer-wisconsin-data.csv (UCI repo).

## Model Architecture (Commonly used)
- Input: Dataset features.
- Hidden: 2 Dense layers (32 neurons, ReLU).
- Output: Dense(1, sigmoid).
- Example: `model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])`.

## Results
- Ovarian cancer dataset: Achieves ~84% accuracy after 15 epochs.
- Breast cancer: ~99% with early stopping, validation accuracy 98.9%.

Built as part of ML engineering practice focusing on neural networks and data science.
