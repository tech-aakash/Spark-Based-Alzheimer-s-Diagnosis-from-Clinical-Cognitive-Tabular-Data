# 01_preproc_numeric.py
from pyspark.sql import functions as F, types as T
from pyspark.ml.feature import Imputer
from utils import make_spark, timer, ensure_dir

# ==== CONFIG (adjust here) ====
INPUT_CSV   = "data/alzheimers_disease_data.csv"
OUT_DIR     = "stage1_numeric"        # parquet dir
ID_COLS     = []                      # e.g., ["PatientID"]
LABEL_COL   = "Diagnosis"
IMPUTE_STRATEGY = "mean"              # "mean" or "median"
# ==============================

spark = make_spark("Stage1_Preproc_Numeric")

with timer("Load CSV"):
    df0 = spark.read.csv(INPUT_CSV, header=True, inferSchema=True)

print("Schema:")
df0.printSchema()
print("Rows:", df0.count())

with timer("Drop duplicates"):
    df = df0.dropDuplicates()

# Identify numeric columns (keep label as-is for now)
numeric_cols = [f.name for f in df.schema.fields
                if isinstance(f.dataType, (T.IntegerType, T.LongType, T.FloatType, T.DoubleType))]

# Basic missing counts
print("Null counts (top 20 columns):")
nulls = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns])
nulls.show(truncate=False)

# Simple range checks (min/max) for numerics
with timer("Min/Max summary"):
    minmax = df.select([F.min(c).alias(f"{c}__min") for c in numeric_cols] +
                       [F.max(c).alias(f"{c}__max") for c in numeric_cols])
    minmax.show(truncate=False)

# Impute numeric nulls in place
with timer(f"Impute numeric ({IMPUTE_STRATEGY})"):
    if numeric_cols:
        imputer = Imputer(strategy=IMPUTE_STRATEGY, inputCols=numeric_cols, outputCols=numeric_cols)
        df = imputer.fit(df).transform(df)

# Write cleaned numeric table
with timer("Write parquet"):
    ensure_dir(OUT_DIR)
    df.write.mode("overwrite").parquet(OUT_DIR)

print("✅ Stage1 complete ->", OUT_DIR)
spark.stop()
