# Credit Card Fraud Detection & Analysis

A machine learning project to detect fraudulent credit card transactions in a highly imbalanced dataset, using Python, Scikit-learn, and SMOTE for class balancing.

## Problem Statement

Credit card fraud is rare but costly — in this dataset, fraudulent transactions make up only **~0.17%** of all transactions. A model that simply predicts "not fraud" every time would be 99.8% accurate but completely useless. The goal of this project was to build a classifier that can actually catch fraudulent transactions while keeping false positives manageable.

## Dataset

- **Source:** [Credit Card Fraud Detection (Kaggle / ULB Machine Learning Group)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions, 31 columns
- **Features:** `Time`, `Amount`, and 28 anonymized PCA-transformed features (`V1`–`V28`)
- **Target:** `Class` (0 = legitimate, 1 = fraud)
- **Class distribution:** 492 fraud cases out of 284,807 total (~0.17%)

## Approach

1. **Exploratory Data Analysis** — examined class imbalance and transaction amount distributions across fraud/non-fraud classes.
2. **Preprocessing** — scaled `Amount` and `Time` using `StandardScaler` (the other features were already PCA-scaled).
3. **Handling Class Imbalance** — applied **SMOTE** (Synthetic Minority Oversampling) on the training set to balance the fraud/non-fraud classes.
4. **Modeling** — trained and compared two classifiers:
   - Logistic Regression (baseline)
   - Random Forest Classifier
5. **Evaluation** — used precision, recall, F1-score, and ROC-AUC instead of accuracy, since accuracy is misleading on imbalanced data.

## Results

| Metric (fraud class) | Logistic Regression | Random Forest |
|---|---|---|
| Precision | 5.81% | 81.44% |
| Recall | 91.84% | 80.61% |
| F1-score | 0.1094 | 0.8103 |
| ROC-AUC | 0.9698 | 0.9688 |

**Random Forest confusion matrix:**

| | Predicted: Legit | Predicted: Fraud |
|---|---|---|
| **Actual: Legit** | 56,846 | 18 |
| **Actual: Fraud** | 19 | 79 |

**Random Forest ROC curve:** AUC = 0.97

### Key Insight

Logistic Regression caught more fraud cases (91.8% recall) but generated a huge number of false positives (only 5.8% precision), which would overwhelm a real fraud-review team. **Random Forest struck a much better balance** — correctly identifying 79 of 98 fraud cases with only 18 false alarms out of 56,864 legitimate transactions, making it the more practical model for deployment.

## Tech Stack

- Python
- Pandas, NumPy — data cleaning and manipulation
- Scikit-learn — modeling and evaluation
- imbalanced-learn (SMOTE) — class balancing
- Matplotlib / Seaborn — visualization

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

```python
# 1. Load data
df = pd.read_csv('creditcard.csv')

# 2. Preprocess (scale Amount/Time, drop originals)
# 3. Train/test split with stratification
# 4. Apply SMOTE on training data
# 5. Train Logistic Regression and Random Forest
# 6. Evaluate with classification_report, recall_score, roc_auc_score
# 7. Visualize confusion matrix and ROC curve
```

See the full notebook (`Credit_Card_Fraud_Detection.ipynb`) for step-by-step code.

## Future Improvements

- Try gradient boosting models (XGBoost, LightGBM) for potentially better precision-recall balance.
- Tune the classification threshold instead of using the default 0.5 to further optimize recall vs. precision trade-off.
- Add cost-sensitive learning, since false negatives (missed fraud) and false positives (blocked legitimate transactions) have different real-world costs.
