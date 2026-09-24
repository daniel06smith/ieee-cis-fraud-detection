from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def get_spark():
    spark = SparkSession.builder \
        .appName("FraudDetection") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_data(spark, trans_path, id_path):
    trans = spark.read.csv(trans_path, header=True, inferSchema=True)
    identity = spark.read.csv(id_path, header=True, inferSchema=True)
    # Join identity info onto transactions (left join — most rows won't have identity)
    df = trans.join(identity, on="TransactionID", how="left")
    return df


def engineer_features(df):
    """
    Three groups of features:
    1. Time features — extract hour/day patterns from TransactionDT
    2. Frequency encoding — how often does each card/email/device appear?
    3. Aggregation features — rolling stats per card (mean amount, transaction count)
    """

    # --- 1. Time features ---
    # TransactionDT is seconds offset from a reference date, not a unix timestamp
    # We extract cyclic time-of-day and day-of-week signals
    df = df.withColumn("hours", (F.col("TransactionDT") / 3600).cast("int") % 24)
    df = df.withColumn("day_of_week", (F.col("TransactionDT") / (3600 * 24)).cast("int") % 7)

    # --- 2. Frequency encoding ---
    # How many times does each card1 value appear in the dataset?
    # Rare cards behave differently from common ones
    for col_name in ["card1", "card2", "P_emaildomain", "R_emaildomain", "DeviceInfo"]:
        freq = df.groupBy(col_name).count().withColumnRenamed("count", f"{col_name}_freq")
        df = df.join(freq, on=col_name, how="left")

    # --- 3. Aggregation features (per card1) ---
    # For each card, compute: mean transaction amount, std, and transaction count
    # This captures "is this transaction unusual for this card?"
    card_window = Window.partitionBy("card1")
    df = df.withColumn("card1_amt_mean", F.mean("TransactionAmt").over(card_window))
    df = df.withColumn("card1_amt_std", F.stddev("TransactionAmt").over(card_window))
    df = df.withColumn("card1_count", F.count("TransactionID").over(card_window))

    # Amount deviation: how far is this transaction from the card's typical amount?
    df = df.withColumn(
        "amt_deviation",
        (F.col("TransactionAmt") - F.col("card1_amt_mean")) / (F.col("card1_amt_std") + 1e-6)
    )

    return df


def run_pipeline(trans_path, id_path, output_path):
    spark = get_spark()
    df = load_data(spark, trans_path, id_path)
    df = engineer_features(df)

    # Write out as parquet — much faster to reload than CSV
    df.write.mode("overwrite").parquet(output_path)
    print(f"Feature engineered data written to {output_path}")
    print(f"Row count: {df.count()}, Column count: {len(df.columns)}")
    spark.stop()


if __name__ == "__main__":
    run_pipeline(
        trans_path="data/raw/train_transaction.csv",
        id_path="data/raw/train_identity.csv",
        output_path="data/processed/train_features.parquet"
    )