import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from src.preprocess import run_preprocessing

def tune_xgboost(X_train, y_train, X_val, y_val):
    """
    Manual grid search over key XGBoost hyperparameters.
    Each combination is logged as a separate MLflow run,
    so you can compare them all in the UI afterward.

    Key params to understand:
    - max_depth: how deep each tree grows (deeper = more complex, more overfit risk)
    - learning_rate: step size per tree (lower = more trees needed but better generalization)
    - colsample_bytree: fraction of features used per tree (lower = more regularization)
    - min_child_weight: minimum samples in a leaf (higher = more regularization)
    """

    param_grid = [
        {"max_depth": 6, "learning_rate": 0.05, "colsample_bytree": 0.4, "min_child_weight": 1},
        {"max_depth": 6, "learning_rate": 0.05, "colsample_bytree": 0.6, "min_child_weight": 5},
        {"max_depth": 8, "learning_rate": 0.05, "colsample_bytree": 0.4, "min_child_weight": 1},
        {"max_depth": 8, "learning_rate": 0.03, "colsample_bytree": 0.4, "min_child_weight": 5},
        {"max_depth": 6, "learning_rate": 0.01, "colsample_bytree": 0.6, "min_child_weight": 5},
    ]

    best_auc = 0
    best_params = None

    mlflow.set_experiment("ieee-fraud-xgb-tuning")

    for i, params in enumerate(param_grid):
        full_params = {
            **params,
            "n_estimators": 500,
            "subsample": 0.8,
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
        }

        run_name = f"xgb_depth{params['max_depth']}_lr{params['learning_rate']}_col{params['colsample_bytree']}"

        print(f"\nRun {i+1}/{len(param_grid)}: {run_name}")

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(full_params)

            model = XGBClassifier(**full_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False   # suppress per-tree output during tuning
            )

            y_prob = model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_prob)
            pr_auc = average_precision_score(y_val, y_prob)

            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("pr_auc", pr_auc)
            mlflow.xgboost.log_model(model, "model")

            print(f"  ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

            if roc_auc > best_auc:
                best_auc = roc_auc
                best_params = full_params
                mlflow.set_tag("best_model", "true")

    print(f"\nBest ROC-AUC: {best_auc:.4f}")
    print(f"Best params: {best_params}")
    return best_params


if __name__ == "__main__":
    print("Loading data...")
    X_train, y_train, X_val, y_val = run_preprocessing(
        "data/processed/train_features.parquet"
    )

    best_params = tune_xgboost(X_train, y_train, X_val, y_val)