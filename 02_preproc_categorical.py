# 02_preproc_categorical.py
# Author: Gideon (Preprocessing Stage 2)
# Task: Handle categorical variables, encode, scale numerics, and assemble final feature vector.

from pyspark.sql import functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
)
from utils import make_spark, timer, ensure_dir

# =======================
# CONFIGURATION SECTION
# =======================
IN_DIR = "stage1_numeric"           # Input parquet from Stage 1
OUT_DIR = "stage2_featurestore"     # Output feature store parquet
LABEL_COL = "Diagnosis"             # Label column for classification
ID_COLS = ["PatientID", "DoctorInCharge"]  # Non-feature identifiers
FILL_CATEG = "Unknown"              # Replacement for missing categorical values
# =======================

spark = make_spark("Stage2_Preproc_Categorical")

# ---------------------------------------------------
# STEP 1: LOAD CLEANED NUMERIC DATA FROM STAGE 1
# ---------------------------------------------------
with timer("Load Stage1 parquet"):
    df = spark.read.parquet(IN_DIR)

print("Schema after Stage 1:")
df.printSchema()
print("Total records:", df.count())

# ---------------------------------------------------
# STEP 2: DEFINE CATEGORICAL COLUMNS (INCLUDING CODED NUMERICS)
# ---------------------------------------------------
categorical_force = [
    "Gender", "Ethnicity", "EducationLevel", "Smoking", "AlcoholConsumption",
    "PhysicalActivity", "DietQuality", "SleepQuality", "FamilyHistoryAlzheimers",
    "CardiovascularDisease", "Diabetes", "Depression", "HeadInjury", "Hypertension",
    "Confusion", "Disorientation", "PersonalityChanges",
    "DifficultyCompletingTasks", "Forgetfulness"
]

# Cast all forced categorical columns to string type
for c in categorical_force:
    if c in df.columns:
        df = df.withColumn(c, F.col(c).cast("string"))

# ---------------------------------------------------
# STEP 3: FILL MISSING VALUES
# ---------------------------------------------------
with timer("Fill categorical nulls"):
    for c in df.columns:
        if c not in ID_COLS + [LABEL_COL]:
            if isinstance(df.schema[c].dataType, T.StringType):
                df = df.withColumn(
                    c, F.when(F.col(c).isNull() | (F.trim(F.col(c)) == ""), FILL_CATEG)
                        .otherwise(F.col(c))
                )

# ---------------------------------------------------
# STEP 4: SPLIT BY TYPE
# ---------------------------------------------------
numeric_cols = [
    f.name for f in df.schema.fields
    if isinstance(f.dataType, (T.IntegerType, T.LongType, T.FloatType, T.DoubleType))
    and f.name not in ID_COLS + [LABEL_COL]
]
categorical_cols = [
    f.name for f in df.schema.fields
    if isinstance(f.dataType, T.StringType)
    and f.name not in ID_COLS + [LABEL_COL]
]

print(f"Numeric columns ({len(numeric_cols)}):", numeric_cols[:10])
print(f"Categorical columns ({len(categorical_cols)}):", categorical_cols[:10])

# ---------------------------------------------------
# STEP 5: BUILD PIPELINE (INDEX, OHE, SCALE, ASSEMBLE)
# ---------------------------------------------------
label_indexer = StringIndexer(
    inputCol=LABEL_COL, outputCol="label", handleInvalid="keep"
)

# Index + One-hot encode categorical columns
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}__idx", handleInvalid="keep")
    for c in categorical_cols
]
ohe = OneHotEncoder(
    inputCols=[f"{c}__idx" for c in categorical_cols],
    outputCols=[f"{c}__vec" for c in categorical_cols],
    handleInvalid="keep"
)

# Assemble and scale numeric columns
num_assembler = VectorAssembler(inputCols=numeric_cols, outputCol="num_vec")
scaler = StandardScaler(inputCol="num_vec", outputCol="num_scaled", withMean=True, withStd=True)

# Final feature assembler
final_assembler = VectorAssembler(
    inputCols=["num_scaled"] + [f"{c}__vec" for c in categorical_cols],
    outputCol="features"
)

# Complete pipeline
pipe = Pipeline(stages=[label_indexer] + indexers + [ohe, num_assembler, scaler, final_assembler])

# ---------------------------------------------------
# STEP 6: FIT + TRANSFORM
# ---------------------------------------------------
with timer("Fit + Transform pipeline"):
    model = pipe.fit(df)
    fs = model.transform(df)

# Keep only useful columns
cols_to_keep = ["features", "label"] + ID_COLS
fs = fs.select(*cols_to_keep)

# ---------------------------------------------------
# STEP 7: SAVE FEATURE STORE
# ---------------------------------------------------
with timer("Write feature store"):
    ensure_dir(OUT_DIR)
    fs.write.mode("overwrite").parquet(OUT_DIR)

print(f"✅ Stage 2 complete! Feature store saved to: {OUT_DIR}")
spark.stop()

