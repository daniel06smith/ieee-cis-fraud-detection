import pandas as pd
import numpy as np
import mlflow
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from pyspark.sql import SparkSession
from src.features import get_spark, engineer_features
from src.preprocess import (
    load_features, drop_high_null_cols,
    encode_categoricals, fill_numerics, split_features_target,
    apply_smote, run_preprocessing
)


def generate_test_features(spark, test_trans_path, test_id_path, output_path):
    """Run the same PySpark feature engineering on the test set."""
    print("Engineering test features with PySpark...")
    test_trans = spark.read.csv(test_trans_path, header=True, inferSchema=True)
    test_id = spark.read.csv(test_id_path, header=True, inferSchema=True)
    df = test_trans.join(test_id, on="TransactionID", how="left")
    df = engineer_features(df)
    df.write.mode("overwrite").parquet(output_path)
    print(f"Test features written to {output_path}")
    spark.stop()


def align_columns(train_df, test_df):
    """
    Ensure test set has exactly the same columns as train, in the same order.
    Columns in train but missing from test get filled with -999.
    Columns in test but not in train get dropped.
    """
    train_cols = [c for c in train_df.columns if c not in ['TransactionID', 'isFraud']]
    for col in train_cols:
        if col not in test_df.columns:
            test_df[col] = -999
    return test_df[train_cols]


def preprocess_test(test_parquet_path, train_parquet_path):
    """
    Apply the same preprocessing steps to test data.
    Critical: fit encoders on train, apply to test — never fit on test data.
    """
    from sklearn.preprocessing import LabelEncoder

    # Load both so we can align columns
    train_df = load_features(train_parquet_path)
    test_df = load_features(test_parquet_path)

    # Save test TransactionIDs for submission file
    test_ids = test_df['TransactionID'].values

    # Drop same high-null columns as train (based on train null rates)
    null_rates = train_df.isnull().mean()
    cols_to_drop = null_rates[null_rates > 0.9].index.tolist()
    train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
    test_df = test_df.drop(columns=cols_to_drop, errors='ignore')

        # Label encode — fit on train, transform both
    train_cat = set(train_df.select_dtypes(include=['object']).columns.tolist())
    test_cat = set(test_df.select_dtypes(include=['object']).columns.tolist())
    cat_cols = list(train_cat & test_cat)  # only columns present in BOTH
    cat_cols = [c for c in cat_cols if c not in ['TransactionID', 'isFraud']]

    # Columns only in train: fill with 0 (already numeric after encoding)
    train_only_cat = train_cat - test_cat - {'TransactionID', 'isFraud'}
    for col in train_only_cat:
        train_df[col] = 0
        test_df[col] = 0

    le = LabelEncoder()
    for col in cat_cols:
        train_df[col] = train_df[col].fillna('missing')
        test_df[col] = test_df[col].fillna('missing')

        # Fit on combined to handle unseen categories in test
        combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))

    # Fill numerics
    train_df = fill_numerics(train_df)
    test_df = fill_numerics(test_df)

    # Align test columns to train
    X_test = align_columns(train_df, test_df)

    # Get train features/target for model fitting
    X_train, y_train = split_features_target(train_df)

    return X_train, y_train, X_test, test_ids


if __name__ == "__main__":
    import os
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("submissions", exist_ok=True)

    # Step 1: PySpark feature engineering on test set
    spark = get_spark()
    generate_test_features(
        spark,
        test_trans_path="data/raw/test_transaction.csv",
        test_id_path="data/raw/test_identity.csv",
        output_path="data/processed/test_features.parquet"
    )

    # Step 2: Preprocess
    print("\nPreprocessing...")
    X_train, y_train, X_test, test_ids = preprocess_test(
        test_parquet_path="data/processed/test_features.parquet",
        train_parquet_path="data/processed/train_features.parquet"
    )

    # Step 3: Apply SMOTE to training data
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    # Step 4: Train final XGBoost on full training data
    print("\nTraining final XGBoost on full training set...")
    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.4, random_state=42,
        n_jobs=-1, eval_metric="auc"
    )
    model.fit(X_train_sm, y_train_sm, verbose=100)

    # Step 5: Generate predictions
    print("\nGenerating predictions...")
    test_probs = model.predict_proba(X_test)[:, 1]

    # Step 6: Write submission file
    submission = pd.DataFrame({
        'TransactionID': test_ids,
        'isFraud': test_probs
    })
    submission_path = "submissions/submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\nSubmission saved to {submission_path}")
    print(f"Shape: {submission.shape}")
    print(submission.head())