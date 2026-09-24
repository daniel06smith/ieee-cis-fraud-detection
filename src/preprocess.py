import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def load_features(parquet_path):
    """
    Read the parquet output from PySpark back into pandas.
    For model training, pandas + sklearn is more ergonomic than PySpark MLlib.
    In production you'd stay in Spark, but this is the standard portfolio pattern.
    """
    df = pd.read_parquet(parquet_path)
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def drop_high_null_cols(df, threshold=0.9):
    """Drop columns where more than `threshold` of values are null."""
    null_rates = df.isnull().mean()
    cols_to_drop = null_rates[null_rates > threshold].index.tolist()
    print(f"Dropping {len(cols_to_drop)} columns with >{threshold*100:.0f}% nulls")
    return df.drop(columns=cols_to_drop)


def encode_categoricals(df):
    """
    Label encode categorical columns.
    XGBoost/LightGBM/CatBoost can all handle label-encoded integers.
    We fill nulls with 'missing' first so the encoder doesn't choke.
    """
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Label encoding {len(cat_cols)} categorical columns")

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].fillna('missing')
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def fill_numerics(df):
    """Fill remaining numeric nulls with -999 — a sentinel value tree models handle well."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(-999)
    return df


def split_features_target(df):
    drop_cols = ['TransactionID', 'isFraud']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['isFraud'].astype(int)
    return X, y


def apply_smote(X_train, y_train, sampling_strategy=0.1, random_state=42):
    """
    SMOTE: Synthetic Minority Over-sampling Technique.
    
    Rather than duplicating fraud examples, SMOTE creates synthetic ones
    by interpolating between existing fraud cases in feature space.
    
    sampling_strategy=0.1 means we'll oversample until fraud is 10% of data
    (up from 3.5%) — enough to help without making the dataset unrealistically balanced.
    """
    print(f"Before SMOTE — fraud: {y_train.sum():,} / {len(y_train):,} ({y_train.mean()*100:.1f}%)")
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE  — fraud: {y_resampled.sum():,} / {len(y_resampled):,} ({y_resampled.mean()*100:.1f}%)")
    return X_resampled, y_resampled


def run_preprocessing(parquet_path):
    from sklearn.model_selection import train_test_split

    df = load_features(parquet_path)
    df = drop_high_null_cols(df, threshold=0.9)
    df = encode_categoricals(df)
    df = fill_numerics(df)

    X, y = split_features_target(df)

    # Time-based split: first 75% of transactions for train, last 25% for validation
    # This is more realistic than random split — you don't train on future data
    split_idx = int(len(X) * 0.75)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\nTrain size: {len(X_train):,} | Val size: {len(X_val):,}")

    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    return X_train_sm, y_train_sm, X_val, y_val


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = run_preprocessing("data/processed/train_features.parquet")
    print("\nPreprocessing complete.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape:   {X_val.shape}")