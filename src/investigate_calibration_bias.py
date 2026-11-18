
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
STATES_FILE = PROCESSED / "first_innings_states.csv"

OUTPUT = MODELS / "calibration_investigation"
OUTPUT.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATA
# ==================================================

print("\nLoading prediction and match data...")

predictions = pd.read_csv(
    PREDICTIONS_FILE,
    low_memory=False
)

states = pd.read_csv(
    STATES_FILE,
    low_memory=False
)

# Keep Random Forest predictions
df = predictions[
    predictions["model"] == "Random Forest"
].copy()

# Obtain match-level context
context_columns = [
    "match_id",
    "comp_name",
    "season",
    "venue_id"
]

context = (
    states[context_columns]
    .drop_duplicates(subset=["match_id"])
)

df = df.merge(
    context,
    on="match_id",
    how="left",
    validate="many_to_one"
)

if df["comp_name"].isna().any():
    raise ValueError("Missing competition context after merging.")

df["competition_type"] = np.where(
    df["comp_name"].str.contains("IPL", case=False, na=False),
    "IPL",
    "T20I"
)

# Calculate prediction errors
df["error"] = (
    df["predicted_final_score"] - df["actual_final_score"]
)

df["absolute_error"] = df["error"].abs()

print("Prediction rows:", len(df))
print("Test matches:", df["match_id"].nunique())


# ==================================================
# 3. COMPETITION-WISE BASELINE
# ==================================================

print("\n" + "=" * 65)
print("BASELINE: IPL VS T20I")
print("=" * 65)

baseline = (
    df.groupby("competition_type")
    .agg(
        matches=("match_id", "nunique"),
        observations=("error", "count"),
        actual_mean=("actual_final_score", "mean"),
        predicted_mean=("predicted_final_score", "mean"),
        bias=("error", "mean"),
        MAE=("absolute_error", "mean"),
        RMSE=("error", lambda x: np.sqrt(np.mean(x ** 2)))
    )
    .reset_index()
)

print(baseline.round(3).to_string(index=False))


# ==================================================
# 4. VENUE-WISE IPL ANALYSIS
# ==================================================

print("\n" + "=" * 65)
print("IPL VENUE-WISE CALIBRATION")
print("=" * 65)

ipl = df[df["competition_type"] == "IPL"].copy()

venue_results = (
    ipl.groupby("venue_id")
    .agg(
        matches=("match_id", "nunique"),
        observations=("error", "count"),
        actual_mean=("actual_final_score", "mean"),
        predicted_mean=("predicted_final_score", "mean"),
        bias=("error", "mean"),
        MAE=("absolute_error", "mean")
    )
    .reset_index()
)

# Show venues with at least 3 test matches
venue_results = venue_results[
    venue_results["matches"] >= 3
].sort_values("bias")

print(venue_results.round(3).to_string(index=False))

venue_results.to_csv(
    OUTPUT / "venue_calibration.csv",
    index=False
)


# ==================================================
# 5. INNINGS-STATE ANALYSIS
# ==================================================

print("\n" + "=" * 65)
print("CALIBRATION BY INNINGS STATE")
print("=" * 65)

df["balls_group"] = pd.cut(
    df["legal_balls_remaining"],
    bins=[0, 30, 60, 90, 120],
    labels=["1-30", "31-60", "61-90", "91-120"],
    include_lowest=True
)

df["wickets_group"] = pd.cut(
    df["wickets_lost"],
    bins=[-1, 1, 3, 5, 10],
    labels=["0-1", "2-3", "4-5", "6-10"]
)

state_results = (
    df.groupby(
        ["competition_type", "balls_group", "wickets_group"],
        observed=True
    )
    .agg(
        observations=("error", "count"),
        matches=("match_id", "nunique"),
        bias=("error", "mean"),
        MAE=("absolute_error", "mean")
    )
    .reset_index()
)

print(state_results.round(3).to_string(index=False))

state_results.to_csv(
    OUTPUT / "innings_state_calibration.csv",
    index=False
)


# ==================================================
# 6. MATCH-LEVEL BIAS
# ==================================================

print("\n" + "=" * 65)
print("MATCH-LEVEL IPL CALIBRATION")
print("=" * 65)

match_results = (
    df.groupby(
        ["competition_type", "match_id"]
    )
    .agg(
        actual_score=("actual_final_score", "first"),
        predicted_score=("predicted_final_score", "mean"),
        match_bias=("error", "mean"),
        match_MAE=("absolute_error", "mean")
    )
    .reset_index()
)

match_summary = (
    match_results.groupby("competition_type")
    .agg(
        matches=("match_id", "nunique"),
        mean_match_bias=("match_bias", "mean"),
        median_match_bias=("match_bias", "median"),
        mean_match_MAE=("match_MAE", "mean"),
        underpredicted_matches=(
            "match_bias",
            lambda x: (x < 0).sum()
        ),
        overpredicted_matches=(
            "match_bias",
            lambda x: (x > 0).sum()
        )
    )
    .reset_index()
)

print(match_summary.round(3).to_string(index=False))

match_results.to_csv(
    OUTPUT / "match_level_calibration.csv",
    index=False
)


# ==================================================
# 7. TRAINING-ONLY COMPETITION CALIBRATION
# ==================================================

print("\n" + "=" * 65)
print("TRAINING-ONLY CALIBRATION CORRECTION")
print("=" * 65)

# Training predictions are not available from the original
# training script. Therefore, do not estimate a correction
# using test outcomes or pretend that a test-set correction
# is a valid independent improvement.

print(
    "Skipped: training-set predictions are not currently saved."
)
print(
    "A valid correction requires out-of-sample or "
    "training-only calibration predictions."
)


# ==================================================
# 8. SAVE SUMMARY
# ==================================================

baseline.to_csv(
    OUTPUT / "competition_baseline.csv",
    index=False
)

match_summary.to_csv(
    OUTPUT / "match_bias_summary.csv",
    index=False
)

print("\n" + "=" * 65)
print("CALIBRATION INVESTIGATION COMPLETE")
print("=" * 65)

print("\nSaved files:")
print("calibration_investigation/competition_baseline.csv")
print("calibration_investigation/venue_calibration.csv")
print("calibration_investigation/innings_state_calibration.csv")
print("calibration_investigation/match_level_calibration.csv")
print("calibration_investigation/match_bias_summary.csv")
