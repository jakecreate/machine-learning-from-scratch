# Machine Learning from Scratch
Implementing ML models and concepts from scratch using NumPy for personal learning purposes.
<p align="center" width="100%">
    <img width="33%" src="./images/numpy.png"> 
</p>

## Navigation
* `notebooks/`: Jupyter Notebooks that walk through the implementation, explanation, and visualization of ML models and concepts.
* `custom_models/`: Python scripts containing the class-based implementations of the machine learning models.
* `data_utils/`: Utility scripts for data loading and preprocessing.
* `images/`: Visualizations and plots generated from the notebooks.

## Visualization Highlight
<p align="center" width="100%">
    <img width=100%" src="./images/data_split.png"> 
</p>
<p align="center" width="100%">
    <img width=100%" src="./images/bias-variance_trade-off.png"> 
</p>



## Current Content Overview
- **Models**:
  - `linear_regression.py`
  - `logistic_regression.py` (with L2 Regularization)

- **Concepts**:
  - `1_fundamentals_of_supervised_learning.ipynb`
    - Regression & Classification
    - Defining hypothesis function
    - Mesuring quality of model's fit using:
      - Loss function
        - squared error
        - cross entropy
      - Cost function
        - MSE
        - Log loss
    - Gradient Descent
      - Cause for divergence
    - Feature Scaling
    - Bias-Variance Trade-off
    - Polynomial Regression
      - Hyperparameter Tuning
    - Evaluating Model Performance
    - Implementation of `LinearRegression()` & `LogisticRegression()`
  - `2_historical_simple_algorithms`
    - Artificial Neuron
    - Perceptron Learning Rule
- Data Utility:
  - Train test split
  - Standard scaling


## Requirements
- Python 3.12.11
- numpy==2.3.1
- matplotlib==3.10.0

## Installation
Pip:
```bash
pip install numpy==2.3.1 matplotlib==3.10.0
```
or Conda:
```bash
conda install numpy=2.3.1 matplotlib=3.10.0
```