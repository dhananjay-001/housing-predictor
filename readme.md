# California Housing and Breast Cancer ML Tasks

Four introductory machine learning tasks completed during my AI & ML internship at Maincrafts Technology. Tasks 1 to 3 use the California Housing dataset (regression). Task 4 uses the Breast Cancer Wisconsin dataset (classification). All data comes from `scikit-learn`, so nothing needs to be downloaded separately.

## Repository contents

| File | What it does |
|---|---|
| `task1_ml_linear_regression.ipynb` / `.py` | Task 1: EDA and baseline Linear Regression |
| `AI_ML_Task2_Model_Comparison.ipynb` | Task 2: feature scaling and comparison of Linear Regression, Ridge and Decision Tree |
| `AI_ML_Task3_Model_Validation.ipynb` | Task 3: overfitting check, 5-fold cross-validation and GridSearchCV tuning |
| `AI_ML_Task4_Classification_Evaluation.ipynb` | Task 4: Logistic Regression and Decision Tree classifiers with confusion matrix, precision, recall and ROC-AUC |

## Results

All regression scores are on a held-out 20% test set (`test_size=0.2`, `random_state=42`).

### Task 1: baseline Linear Regression
MAE 0.533 (about $53,300), RMSE 0.745, R² 0.576. Median income (`MedInc`) is the strongest predictor (correlation about 0.69). Target values are capped at 5.0 ($500,000), which makes the model under-predict expensive districts.

### Task 2: model comparison

| Model | Test RMSE | Test R² |
|---|---|---|
| Linear Regression | 0.7456 | 0.5758 |
| Ridge (alpha=1.0) | 0.7456 | 0.5758 |
| Decision Tree (max_depth=5) | **0.7242** | **0.5997** |

### Task 3: overfitting and tuning
An unconstrained Decision Tree memorises the training data (train RMSE 0.0000, test RMSE 0.7037). 5-fold cross-validation gave a mean RMSE of 0.7333. GridSearchCV over `max_depth`, `min_samples_split` and `min_samples_leaf` (140 combinations) chose `max_depth=12`, `min_samples_leaf=5`, `min_samples_split=50`, giving train RMSE 0.4943 and **test RMSE 0.6040**.

### Task 4: breast cancer classification
Stratified 80/20 split (114 test samples: 42 malignant, 72 benign).

| Model | Accuracy | Malignant precision | Malignant recall |
|---|---|---|---|
| Logistic Regression | **0.98** | 0.98 | 0.98 |
| Logistic Regression (`class_weight="balanced"`) | 0.96 | 0.91 | 0.98 |
| Decision Tree | 0.91 | 0.85 | 0.93 |

The baseline Logistic Regression misses 1 malignant tumour and raises 1 false alarm. Its ROC-AUC is 1.00 (rounded to two decimals in the plot legend).

## How to run

```bash
python -m venv ml_env
ml_env\Scripts\activate        # Windows (use: source ml_env/bin/activate on Linux/macOS)
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

Use scikit-learn 1.6 or newer, because the notebooks use `root_mean_squared_error`. Open any notebook and choose **Run All**.

## Tech stack

Python, Jupyter Notebook, pandas, NumPy, scikit-learn, Matplotlib, Seaborn.

## Author

Dhananjay Sahu, B.Tech CSE, Gurugram University.
