#!/bin/bash
# =============================================================
# Alzheimer's Spark Pipeline Runner (Timing Only)
# =============================================================

echo "🚀 Starting full Spark pipeline on $(date)"
echo "============================================================"

overall_start=$(date +%s)

run_stage() {
    local file=$1
    local name=$2

    echo "▶️  Running $name ($file)"
    local start=$(date +%s)

    # Run silently — only capture timing (hide Spark logs)
    spark-submit --master spark://hadoop1:7077 "$file" > /dev/null 2>&1

    local end=$(date +%s)
    local elapsed=$((end - start))
    echo "✅ $name completed in ${elapsed}s"
    echo "------------------------------------------------------------"
}

run_stage 01_preproc_numeric.py       "Stage 1 - Numeric Preprocessing"
run_stage 02_preproc_categorical.py   "Stage 2 - Categorical Preprocessing"
run_stage 03_stats_and_lr.py          "Stage 3 - Logistic Regression"
run_stage 04_rf_and_feature_select.py "Stage 4 - Random Forest"

overall_end=$(date +%s)
total_elapsed=$((overall_end - overall_start))

echo "============================================================"
echo "✅ All stages completed successfully."
echo "🕒 Total pipeline time: ${total_elapsed}s (~$(echo "scale=2; $total_elapsed/60" | bc) minutes)"
echo "🏁 Finished at $(date)"
echo "============================================================"

