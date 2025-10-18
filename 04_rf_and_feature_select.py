#!/usr/bin/env python3
from pyspark.sql import functions as F
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import ChiSqSelector
from utils import make_spark, timer

# ----------------------------------------------------------------------------
# Initialize Spark
# ----------------------------------------------------------------------------
spark = make_spark("Stage4_RF_and_FeatureSelect")

# ----------------------------------------------------------------------------
# Load feature store
# ----------------------------------------------------------------------------
INPUT_PATH = "stage2_featurestore"
with timer("Load feature store"):
    fs = spark.read.parquet(INPUT_PATH)
    fs = fs.withColumn("label", F.col("label").cast("double"))
    fs = fs.filter(F.col("label").isin([0.0, 1.0]))  # 🔹 enforce binary
    print("✅ Filtered dataset to binary labels (0.0 and 1.0 only).")
    print(f"Total records: {fs.count()}")
    fs.printSchema()

# ----------------------------------------------------------------------------
# Feature Selection: ChiSqSelector
# ----------------------------------------------------------------------------
with timer("ChiSqSelector top 50"):
    selector = ChiSqSelector(
        numTopFeatures=50,
        featuresCol="features",
        outputCol="selectedFeatures",
        labelCol="label"
    )
    fs_selected = selector.fit(fs).transform(fs)

# ----------------------------------------------------------------------------
# Train/Test Split
# ----------------------------------------------------------------------------
train, test = fs_selected.randomSplit([0.8, 0.2], seed=42)

# ----------------------------------------------------------------------------
# Random Forest Classifier
# ----------------------------------------------------------------------------
with timer("Fit RandomForest"):
    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="selectedFeatures",
        numTrees=100,
        maxDepth=10,
        seed=42,
        probabilityCol="probability",
        rawPredictionCol="rawPrediction"
    )
    model = rf.fit(train)
    pred = model.transform(test)
    pred.select("label", "prediction", "probability").show(10, truncate=False)

# ----------------------------------------------------------------------------
# Evaluate
# ----------------------------------------------------------------------------
with timer("Evaluate RF"):
    eval_acc = MulticlassClassificationEvaluator(metricName="accuracy", labelCol="label", predictionCol="prediction")
    eval_f1 = MulticlassClassificationEvaluator(metricName="f1", labelCol="label", predictionCol="prediction")
    eval_auc = BinaryClassificationEvaluator(metricName="areaUnderROC", labelCol="label", rawPredictionCol="rawPrediction")

    acc = eval_acc.evaluate(pred)
    f1 = eval_f1.evaluate(pred)
    auc = eval_auc.evaluate(pred)

    print("✅ Random Forest Metrics:")
    print(f"   Accuracy = {acc:.4f}")
    print(f"   F1-score = {f1:.4f}")
    print(f"   AUC      = {auc:.4f}")

# ----------------------------------------------------------------------------
# Feature Importances
# ----------------------------------------------------------------------------
with timer("Feature Importances"):
    importances = model.featureImportances
    top_features = importances.toArray().tolist()
    print("Top 10 feature importances (ChiSq-selected):")
    for idx, score in enumerate(top_features[:10]):
        print(f"   Feature_{idx}: {score:.5f}")

# ----------------------------------------------------------------------------
# Stop Spark
# ----------------------------------------------------------------------------
spark.stop()
print("✅ Stage 4 complete! Random Forest + Feature Selection done.")

