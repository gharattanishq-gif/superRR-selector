
import pandas as pd
import numpy as np
from pathlib import Path


# ==================================================
# 1. PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

PROCESSED = ROOT / "outputs" / "processed"
MODELS = ROOT / "outputs" / "models"

PREDICTIONS_FILE = MODELS / "first_innings_test_predictions.csv"
MATCHES_FILE = PROCESSED / "first_innings_states.csv"

MODELS.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATA
# ==================================================

print("\nLoading model predictions and match context...")

predictions = pd.read_csv(
    PREDICTIONS_FILE,
    low_memory=False
)

states = pd.read_csv(
    MATCHES_FILE,
    usecols=["match_id", "comp_name", "season"],
    low_memory=False
)

# Retain Random Forest predictions only
df = predictions[
    predictions["model"] == "Random Forest"
].copy()

# Attach competition and season using match_id
match_context = states.drop_duplicates(
    subset=["match_id"]
)

df = df.merge(
    match_context,
    on="match_id",
    how="left",
    validate="many_to_one"
)

if df["comp_name"].isna().any():
    raise ValueError("Some predictions could not be matched to competition context.")

print("Random Forest prediction rows:", len(df))
print("Test matches:", df["match_id"].nunique())


# ==================================================
# 3. CALCULATE ERRORS
# ==================================================

df["error"] = (
    df["predicted_final_score"] - df["actual_final_score"]
)

df["absolute_error"] = df["error"].abs()


# ==================================================
# 4. CALIBRATION BY PREDICTED SCORE RANGE
# ==================================================

score_bins = [0, 100, 125, 150, 175, 200, 225, 350]

score_labels = [
    "0-100",
    "101-125",
    "126-150",
    "151-175",
    "176-200",
    "201-225",
    "226+"
]

df["predicted_score_range"] = pd.cut(
    df["predicted_final_score"],
    bins=score_bins,
    labels=score_labels,
    include_lowest=True
)

score_calibration = (
    df.groupby("predicted_score_range", observed=True)
    .agg(
        observations=("error", "count"),
        actual_mean=("actual_final_score", "mean"),
        predicted_mean=("predicted_final_score", "mean"),
        mean_error=("error", "mean"),
        MAE=("absolute_error", "mean")
    )
    .reset_index()
)

score_calibration["calibration_gap"] = (
    score_calibration["actual_mean"]
    - score_calibration["predicted_mean"]
)


# ==================================================
# 5. CALIBRATION BY COMPETITION
# ==================================================

competition_calibration = (
    df.groupby("comp_name")
    .agg(
        matches=("match_id", "nunique"),
        observations=("error", "count"),
        actual_mean=("actual_final_score", "mean"),
        predicted_mean=("predicted_final_score", "mean"),
        mean_error=("error", "mean"),
        MAE=("absolute_error", "mean")
    )
    .reset_index()
)

competition_calibration["calibration_gap"] = (
    competition_calibration["actual_mean"]
    - competition_calibration["predicted_mean"]
)


# ==================================================
# 6. IPL VS T20I CALIBRATION
# ==================================================

df["competition_type"] = np.where(
    df["comp_name"].str.contains("IPL", case=False, na=False),
    "IPL",
    "T20I"
)

type_calibration = (
    df.groupby("competition_type")
    .agg(
        matches=("match_id", "nunique"),
        observations=("error", "count"),
        actual_mean=("actual_final_score", "mean"),
        predicted_mean=("predicted_final_score", "mean"),
        mean_error=("error", "mean"),
        MAE=("absolute_error", "mean"),
        RMSE=("error", lambda x: np.sqrt(np.mean(x ** 2)))
    )
    .reset_index()
)

type_calibration["calibration_gap"] = (
    type_calibration["actual_mean"]
    - type_calibration["predicted_mean"]
)


# ==================================================
# 7. SEASON-WISE CALIBRATION
# ==================================================

season_calibration = (
    df.groupby("season")
    .agg(
        matches=("match_id", "nunique"),
        observations=("error", "count"),
        actual_mean=("actual_final_score", "mean"),
        predicted_mean=("predicted_final_score", "mean"),
        mean_error=("error", "mean"),
        MAE=("absolute_error", "mean")
    )
    .reset_index()
)

season_calibration["calibration_gap"] = (
    season_calibration["actual_mean"]
    - season_calibration["predicted_mean"]
)


# ==================================================
# 8. CALIBRATION BY INNINGS STAGE
# ==================================================

stage_bins = [0, 12, 30, 60, 90, 120]

stage_labels = [
    "1-12 balls",
    "13-30 balls",
    "31-60 balls",
    "61-90 balls",
    "91-120 balls"
]

df["innings_stage"] = pd.cut(
    df["legal_balls_remaining"],
    bins=stage_bins,
    labels=stage_labels,
    include_lowest=True
)

stage_calibration = (
    df.groupby("innings_stage", observed=True)
    .agg(
        observations=("error", "count"),
        actual_mean=("actual_final_score", "mean"),
        predicted_mean=("predicted_final_score", "mean"),
        mean_error=("error", "mean"),
        MAE=("absolute_error", "mean")
    )
    .reset_index()
)

stage_calibration["calibration_gap"] = (
    stage_calibration["actual_mean"]
    - stage_calibration["predicted_mean"]
)


# ==================================================
# 9. SAVE CALIBRATION RESULTS
# ==================================================

outputs = {
    "score_range_calibration.csv": score_calibration,
    "competition_calibration.csv": competition_calibration,
    "competition_type_calibration.csv": type_calibration,
    "season_calibration.csv": season_calibration,
    "innings_stage_calibration.csv": stage_calibration
}

for filename, result in outputs.items():
    result.to_csv(MODELS / filename, index=False)


# ==================================================
# 10. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 65)
print("CALIBRATION BY PREDICTED SCORE RANGE")
print("=" * 65)
print(score_calibration.round(3).to_string(index=False))

print("\n" + "=" * 65)
print("CALIBRATION BY COMPETITION TYPE")
print("=" * 65)
print(type_calibration.round(3).to_string(index=False))

print("\n" + "=" * 65)
print("SEASON-WISE CALIBRATION")
print("=" * 65)
print(season_calibration.round(3).to_string(index=False))

print("\n" + "=" * 65)
print("CALIBRATION BY INNINGS STAGE")
print("=" * 65)
print(stage_calibration.round(3).to_string(index=False))

print("\n" + "=" * 65)
print("CALIBRATION ANALYSIS COMPLETE")
print("=" * 65)

print("\nSaved files:")
for filename in outputs:
    print(filename)
