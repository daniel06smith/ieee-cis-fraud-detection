import numpy as np
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    precision_recall_curve
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from src.preprocess import run_preprocessing


def get_predictions(X_train, y_train, X_val, y_val):
    """Retrain all three models and return their val probabilities."""

    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.4, random_state=42,
        n_jobs=-1, eval_metric="auc"
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_prob = xgb.predict_proba(X_val)[:, 1]

    print("Training LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.4, random_state=42,
        n_jobs=-1, verbose=-1
    )
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    lgbm_prob = lgbm.predict_proba(X_val)[:, 1]

    print("Training CatBoost...")
    cb = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        random_seed=42, verbose=False
    )
    cb.fit(X_train, y_train, eval_set=(X_val, y_val))
    cb_prob = cb.predict_proba(X_val)[:, 1]

    return xgb, lgbm, cb, xgb_prob, lgbm_prob, cb_prob


def find_best_weights(xgb_prob, lgbm_prob, cb_prob, y_val):
    """
    Grid search over ensemble weights.
    Instead of equal weighting (1/3 each), we try different
    combinations and pick the one with the best ROC-AUC.
    """
    best_auc = 0
    best_weights = (1/3, 1/3, 1/3)

    # Try weight combinations that sum to 1
    step = 0.1
    weights = np.arange(0, 1.1, step).round(1)

    print("Searching for best ensemble weights...")
    for w1 in weights:
        for w2 in weights:
            w3 = round(1 - w1 - w2, 1)
            if w3 < 0:
                continue
            ensemble = w1 * xgb_prob + w2 * lgbm_prob + w3 * cb_prob
            auc = roc_auc_score(y_val, ensemble)
            if auc > best_auc:
                best_auc = auc
                best_weights = (w1, w2, w3)

    print(f"Best weights — XGB: {best_weights[0]}, LGBM: {best_weights[1]}, CB: {best_weights[2]}")
    print(f"Best weighted ensemble ROC-AUC: {best_auc:.4f}")
    return best_weights, best_auc


def optimize_threshold(ensemble_prob, y_val):
    """
    By default, classifiers use 0.5 as the decision threshold.
    For imbalanced problems this is rarely optimal.
    
    We sweep thresholds and pick the one maximising F1,
    which balances precision (don't cry wolf) and recall (catch fraud).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val, ensemble_prob)

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"\nDefault threshold (0.5):")
    default_preds = (ensemble_prob >= 0.5).astype(int)
    print(f"  Precision: {precision_score(y_val, default_preds):.4f}")
    print(f"  Recall:    {recall_score(y_val, default_preds):.4f}")
    print(f"  F1:        {f1_score(y_val, default_preds):.4f}")

    print(f"\nOptimized threshold ({best_threshold:.3f}):")
    opt_preds = (ensemble_prob >= best_threshold).astype(int)
    print(f"  Precision: {precision_score(y_val, opt_preds):.4f}")
    print(f"  Recall:    {recall_score(y_val, opt_preds):.4f}")
    print(f"  F1:        {f1_score(y_val, opt_preds):.4f}")

    return best_threshold


if __name__ == "__main__":
    mlflow.set_experiment("ieee-fraud-ensemble")

    print("Loading data...")
    X_train, y_train, X_val, y_val = run_preprocessing(
        "data/processed/train_features.parquet"
    )

    xgb, lgbm, cb, xgb_prob, lgbm_prob, cb_prob = get_predictions(
        X_train, y_train, X_val, y_val
    )

    best_weights, best_auc = find_best_weights(
        xgb_prob, lgbm_prob, cb_prob, y_val
    )

    ensemble_prob = (
        best_weights[0] * xgb_prob +
        best_weights[1] * lgbm_prob +
        best_weights[2] * cb_prob
    )

    best_threshold = optimize_threshold(ensemble_prob, y_val)

    # Log everything to MLflow
    with mlflow.start_run(run_name="weighted_ensemble_optimized"):
        mlflow.log_param("xgb_weight", best_weights[0])
        mlflow.log_param("lgbm_weight", best_weights[1])
        mlflow.log_param("cb_weight", best_weights[2])
        mlflow.log_param("threshold", round(best_threshold, 3))

        final_preds = (ensemble_prob >= best_threshold).astype(int)
        mlflow.log_metric("roc_auc", best_auc)
        mlflow.log_metric("pr_auc", average_precision_score(y_val, ensemble_prob))
        mlflow.log_metric("f1", f1_score(y_val, final_preds))
        mlflow.log_metric("precision", precision_score(y_val, final_preds))
        mlflow.log_metric("recall", recall_score(y_val, final_preds))

    print(f"\nFinal optimized ensemble ROC-AUC: {best_auc:.4f}")
    print("\nDone. Check MLflow UI for full results.")