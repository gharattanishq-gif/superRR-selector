
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT = (
    ROOT / "outputs" / "models"
    / "final_calibration_experiment"
    / "training_oof_predictions.csv"
)

OUTPUT = (
    ROOT / "outputs" / "models"
    / "final_calibration_experiment"
)

df = pd.read_csv(INPUT, low_memory=False)

df["error"] = (
    df["predicted_final_score"]
    - df["actual_final_score"]
)

# Estimate corrections independently for each chronological fold
stability = (
    df.groupby(["fold", "competition_type"])
    .agg(
        matches=("match_id", "nunique"),
        observations=("error", "count"),
        estimated_bias=("error", "mean")
    )
    .reset_index()
)

stability["correction"] = -stability["estimated_bias"]

print("\nCORRECTION ESTIMATES BY CHRONOLOGICAL FOLD")
print(stability.round(3).to_string(index=False))

# Overall correction stability
summary = (
    stability.groupby("competition_type")
    .agg(
        folds=("fold", "nunique"),
        mean_correction=("correction", "mean"),
        std_correction=("correction", "std"),
        min_correction=("correction", "min"),
        max_correction=("correction", "max")
    )
    .reset_index()
)

print("\nCORRECTION STABILITY SUMMARY")
print(summary.round(3).to_string(index=False))

stability.to_csv(
    OUTPUT / "fold_calibration_stability.csv",
    index=False
)

summary.to_csv(
    OUTPUT / "calibration_stability_summary.csv",
    index=False
)

print("\nStability analysis complete.")
