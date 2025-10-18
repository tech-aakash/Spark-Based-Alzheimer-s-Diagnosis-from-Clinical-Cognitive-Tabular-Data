#!/usr/bin/env python3
from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from utils import timer, make_spark

# ----------------------------------------------------------------------------
# Initialize Spark
# ----------------------------------------------------------------------------
spark = make_spark("Stage3_Stats_and_LR")

# ----------------------------------------------------------------------------
# Load feature store
# ----------------------------------------------------------------------------
INPUT_PATH = "stage2_featurestore"
with timer("Load feature store"):
    fs = spark.read.parquet(INPUT_PATH)
    print("Schema of loaded feature store:")
    fs.printSchema()
    print("Row count:", fs.count())

# ----------------------------------------------------------------------------
# Descriptive stats by label
# ----------------------------------------------------------------------------
with timer("Descriptives by label"):
    label_stats = fs.groupBy("label").count()
    label_stats.show()

# ----------------------------------------------------------------------------
# Clean & enforce binary labels
# ----------------------------------------------------------------------------
with timer("Clean and validate labels"):
    # Force numeric binary labels only
    fs = fs.withColumn("label", F.col("label").cast("double"))
    fs = fs.filter(F.col("label").isin([0.0, 1.0]))
    unique_labels = [row[0] for row in fs.select("label").distinct().collect()]
    print(f"✅ Unique labels after filtering: {unique_labels}")
    print(f"✅ Final record count: {fs.count()}")

# ----------------------------------------------------------------------------
# Logistic Regression model
# ----------------------------------------------------------------------------
with timer("Logistic Regression training"):
    # Split dataset
    train, test = fs.randomSplit([0.8, 0.2], seed=42)

    # Explicitly force binary logistic regression
    lr = LogisticRegression(
        labelCol="label",
        featuresCol="features",
        maxIter=50,
        family="binomial",   # <-- This ensures only 2 classes are modeled
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction"
    )

    model = lr.fit(train)
    preds = model.transform(test)
    preds.select("label", "prediction", "probability").show(10, truncate=False)

# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
with timer("Model evaluation"):
    evaluator_acc = MulticlassClassificationEvaluator(metricName="accuracy", labelCol="label", predictionCol="prediction")
    evaluator_f1 = MulticlassClassificationEvaluator(metricName="f1", labelCol="label", predictionCol="prediction")
    evaluator_auc = BinaryClassificationEvaluator(metricName="areaUnderROC", labelCol="label", rawPredictionCol="rawPrediction")

    acc = evaluator_acc.evaluate(preds)
    f1 = evaluator_f1.evaluate(preds)
    auc = evaluator_auc.evaluate(preds)

    print(f"✅ Logistic Regression metrics:")
    print(f"   Accuracy = {acc:.4f}")
    print(f"   F1-score = {f1:.4f}")
    print(f"   AUC      = {auc:.4f}")

# ----------------------------------------------------------------------------
# Stop Spark
# ----------------------------------------------------------------------------
spark.stop()
print("✅ Stage 3 complete! Logistic Regression training and evaluation done.")

