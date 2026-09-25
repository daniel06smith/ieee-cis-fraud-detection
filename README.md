# IEEE-CIS Fraud Detection

Reproduction of the Kaggle IEEE-CIS Fraud Detection winning solution, built as a
demonstration of production-style ML engineering with PySpark, an XGBoost/LightGBM/CatBoost
ensemble, and MLflow experiment tracking.

## Results

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| XGBoost | 0.8773 | 0.5322 |
| LightGBM | 0.8742 | 0.5200 |
| CatBoost | 0.8654 | 0.4942 |
| Weighted Ensemble | 0.8773 | 0.5322 |

**Threshold-optimized F1: 0.530** (threshold = 0.255, vs 0.481 at default 0.5)
**Kaggle Leaderboard:** Public AUC 0.9225 | Private AUC 0.9066

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) —
590,540 transactions, 394 features, 3.5% fraud rate (severe class imbalance).

## Project Structure
fraud-detection/
├── data/
│ ├── raw/ # Kaggle CSVs (not tracked in git)
│ └── processed/ # PySpark parquet output (not tracked in git)
├── notebooks/
│ └── 01_eda.ipynb # Exploratory data analysis
├── src/
│ ├── features.py # PySpark feature engineering pipeline
│ ├── preprocess.py # Null handling, encoding, SMOTE
│ ├── train.py # Baseline model training with MLflow logging
│ ├── tune.py # XGBoost hyperparameter tuning (5 runs)
│ └── ensemble.py # Weighted ensemble + threshold optimization
├── requirements.txt
└── README.md


## Approach

### 1. Feature Engineering (PySpark)
- **Time features:** hour-of-day and day-of-week extracted from transaction timestamp offset
- **Frequency encoding:** per-card, per-email-domain, and per-device transaction counts
- **Aggregation features:** per-card mean/std transaction amount and deviation score
- Output written as Parquet for efficient downstream loading

### 2. Preprocessing
- Columns with >90% null rate dropped (12 columns removed)
- Categorical columns label-encoded (29 columns)
- Numeric nulls filled with -999 (sentinel value handled natively by tree models)
- Time-based train/val split (75/25) to prevent future data leakage
- SMOTE applied to training set only: fraud rate increased from 3.6% → 9.1%

### 3. Model Training
Three gradient boosting models trained independently and logged to MLflow:
- XGBoost, LightGBM, CatBoost (500 estimators, depth 6, lr 0.05)
- Each run tracked with hyperparameters, ROC-AUC, and PR-AUC

### 4. Hyperparameter Tuning
5-run grid search over XGBoost depth, learning rate, and column subsampling,
with each run logged as a separate MLflow experiment for comparison.

### 5. Ensemble + Threshold Optimization
- Grid search over XGB/LGBM/CatBoost blend weights — XGBoost alone optimal
- Precision-recall curve used to find F1-maximizing threshold (0.255 vs default 0.5)
- Lowering threshold improves recall from 0.336 → 0.438 at cost of precision

## Setup

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/ieee-cis-fraud.git
cd ieee-cis-fraud
pip install -r requirements.txt

# Download data (requires Kaggle API token)
kaggle competitions download -c ieee-fraud-detection -p data/raw/
cd data/raw && unzip ieee-fraud-detection.zip && cd ../..

# Run pipeline
python src/features.py          # PySpark feature engineering
python -m src.preprocess        # Preprocessing check
python -m src.train             # Train all three models
python -m src.tune              # Hyperparameter tuning
python -m src.ensemble          # Weighted ensemble + threshold optimization

# View experiment results
mlflow ui --workers 1 --port 5001
```

## Key Design Decisions

**Why PySpark for feature engineering?**
The transaction dataset is large enough to benefit from distributed-style processing,
and PySpark's DataFrame API mirrors what you'd use on a real data engineering team.
Running in local mode keeps the setup simple while using the same API as a cluster.

**Why SMOTE over class weighting?**
SMOTE creates synthetic minority samples in feature space rather than just duplicating
existing ones, giving gradient boosting models more varied fraud patterns to learn from.
`sampling_strategy=0.1` was chosen to help without making the class balance unrealistic.

**Why threshold optimization?**
In fraud detection, missing a fraudulent transaction is typically far more costly than
a false positive. The default 0.5 threshold optimizes accuracy; lowering it to 0.255
shifts the tradeoff toward recall, which better reflects real-world priorities.