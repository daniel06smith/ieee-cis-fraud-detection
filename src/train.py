import numpy as np
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Import our preprocessing pipeline
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import run_preprocessing


def evaluate(model, X_val, y_val):
    """
    Two metrics matter for imbalanced fraud detection:
    - ROC-AUC: overall ranking quality (competition metric)
    - PR-AUC (Average Precision): better captures performance on the minority class
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)
    return roc_auc, pr_auc, y_prob


def train_xgboost(X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.4,
        "scale_pos_weight": 1,   # we handled imbalance via SMOTE, so keep this at 1
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name="xgboost_baseline"):
        mlflow.log_params(params)
        mlflow.log_param("model_type", "XGBoost")

        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )

        roc_auc, pr_auc, y_prob = evaluate(model, X_val, y_val)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.xgboost.log_model(model, "model")

        print(f"\nXGBoost — ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
        return model, y_prob


def train_lightgbm(X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.4,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    with mlflow.start_run(run_name="lightgbm_baseline"):
        mlflow.log_params(params)
        mlflow.log_param("model_type", "LightGBM")

        model = LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        roc_auc, pr_auc, y_prob = evaluate(model, X_val, y_val)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.lightgbm.log_model(model, "model")

        print(f"\nLightGBM — ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
        return model, y_prob


def train_catboost(X_train, y_train, X_val, y_val):
    params = {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "random_seed": 42,
        "verbose": 100,
    }

    with mlflow.start_run(run_name="catboost_baseline"):
        mlflow.log_params(params)
        mlflow.log_param("model_type", "CatBoost")

        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        roc_auc, pr_auc, y_prob = evaluate(model, X_val, y_val)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.catboost.log_model(model, "model")

        print(f"\nCatBoost — ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
        return model, y_prob


if __name__ == "__main__":
    # Set up MLflow experiment — all runs go under this name
    mlflow.set_experiment("ieee-fraud-detection")

    print("Loading and preprocessing data...")
    X_train, y_train, X_val, y_val = run_preprocessing(
        "data/processed/train_features.parquet"
    )

    print("\n--- Training XGBoost ---")
    xgb_model, xgb_probs = train_xgboost(X_train, y_train, X_val, y_val)

    print("\n--- Training LightGBM ---")
    lgbm_model, lgbm_probs = train_lightgbm(X_train, y_train, X_val, y_val)

    print("\n--- Training CatBoost ---")
    cb_model, cb_probs = train_catboost(X_train, y_train, X_val, y_val)

    # Simple average ensemble
    ensemble_probs = (xgb_probs + lgbm_probs + cb_probs) / 3
    ensemble_auc = roc_auc_score(y_val, ensemble_probs)
    ensemble_pr = average_precision_score(y_val, ensemble_probs)

    with mlflow.start_run(run_name="ensemble_average"):
        mlflow.log_metric("roc_auc", ensemble_auc)
        mlflow.log_metric("pr_auc", ensemble_pr)
        mlflow.log_param("model_type", "Ensemble (XGB+LGBM+CB average)")

    print(f"\n--- Ensemble — ROC-AUC: {ensemble_auc:.4f} | PR-AUC: {ensemble_pr:.4f}")
    print("\nDone. Run `mlflow ui` to view experiment results.")