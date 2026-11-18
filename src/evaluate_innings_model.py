
import pandas as pd
import numpy as np
from pathlib import Path


# ==================================================
# 1. PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "outputs" / "models"

INPUT = MODELS / "first_innings_test_predictions.csv"


# ==================================================
# 2. LOAD PREDICTIONS
# ==================================================

print("\nLoading test predictions...")

df = pd.read_csv(INPUT, low_memory=False)

required_columns = [
    "match_id",
    "legal_balls_remaining",
    "actual_final_score",
    "predicted_final_score",
    "model"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}. "
        "Run train_first_innings_model.py first."
    )

df = df[
    df["model"].isin(["Random Forest", "Extra Trees"])
].copy()

if df.empty:
    raise ValueError("No Random Forest or Extra Trees predictions found.")


# ==================================================
# 3. CALCULATE PREDICTION ERRORS
# ==================================================

df["error"] = (
    df["predicted_final_score"]
    - df["actual_final_score"]
)

df["absolute_error"] = df["error"].abs()


# ==================================================
# 4. CLASSIFY INNINGS STAGES
# ==================================================

bins = [0, 12, 30, 60, 90, 120]

labels = [
    "1-12 balls",
    "13-30 balls",
    "31-60 balls",
    "61-90 balls",
    "91-120 balls"
]

df["innings_stage"] = pd.cut(
    df["legal_balls_remaining"],
    bins=bins,
    labels=labels,
    include_lowest=True
)


# ==================================================
# 5. STAGE-WISE PERFORMANCE
# ==================================================

stage_results = (
    df.groupby(
        ["model", "innings_stage"],
        observed=True
    )
    .agg(
        observations=("absolute_error", "count"),
        MAE=("absolute_error", "mean"),
        RMSE=(
            "error",
            lambda x: np.sqrt(np.mean(x ** 2))
        ),
        Mean_Error=("error", "mean")
    )
    .reset_index()
)


# ==================================================
# 6. MATCH-LEVEL PERFORMANCE
# ==================================================

match_results = (
    df.groupby(["model", "match_id"])
    .agg(
        Match_MAE=("absolute_error", "mean"),
        Mean_Error=("error", "mean")
    )
    .reset_index()
)

overall_match_results = (
    match_results.groupby("model")
    .agg(
        Matches=("match_id", "nunique"),
        Mean_Match_MAE=("Match_MAE", "mean"),
        Mean_Match_Error=("Mean_Error", "mean"),
        Median_Match_MAE=("Match_MAE", "median")
    )
    .reset_index()
)


# ==================================================
# 7. SAVE RESULTS
# ==================================================

stage_results.to_csv(
    MODELS / "innings_stage_evaluation.csv",
    index=False
)

match_results.to_csv(
    MODELS / "match_level_evaluation.csv",
    index=False
)

overall_match_results.to_csv(
    MODELS / "overall_match_level_evaluation.csv",
    index=False
)


# ==================================================
# 8. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 65)
print("PERFORMANCE BY INNINGS STAGE")
print("=" * 65)

print(stage_results.round(3).to_string(index=False))

print("\n" + "=" * 65)
print("OVERALL MATCH-LEVEL PERFORMANCE")
print("=" * 65)

print(overall_match_results.round(3).to_string(index=False))

print("\n" + "=" * 65)
print("EVALUATION COMPLETE")
print("=" * 65)

print("\nFILES SAVED:")
print("innings_stage_evaluation.csv")
print("match_level_evaluation.csv")
print("overall_match_level_evaluation.csv")
