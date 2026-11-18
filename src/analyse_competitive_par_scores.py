
import pandas as pd
import numpy as np
from pathlib import Path


# ==================================================
# 1. PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT = (
    ROOT / "outputs" / "processed"
    / "first_innings_states.csv"
)

OUTPUT = ROOT / "outputs" / "analysis" / "par_scores"
OUTPUT.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATA
# ==================================================

print("\nLoading first-innings states...")

df = pd.read_csv(INPUT, low_memory=False)

required_columns = [
    "match_id",
    "final_score",
    "comp_name",
    "season",
    "venue_id",
    "match_date"
]

missing = [
    column for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["match_date"] = pd.to_datetime(
    df["match_date"], errors="coerce"
)

df["final_score"] = pd.to_numeric(
    df["final_score"], errors="coerce"
)

df["season"] = df["season"].astype(str)
df["venue_id"] = df["venue_id"].astype(str)

df["competition_type"] = np.where(
    df["comp_name"].str.contains("IPL", case=False, na=False),
    "IPL",
    "T20I"
)

print("State rows:", len(df))
print("Unique matches:", df["match_id"].nunique())


# ==================================================
# 3. CREATE ONE RECORD PER MATCH
# ==================================================

print("\nCreating match-level first-innings dataset...")

# Check that each match has only one final score
score_counts = (
    df.groupby("match_id")["final_score"]
    .nunique()
)

invalid_matches = score_counts[score_counts != 1]

if len(invalid_matches) > 0:
    raise ValueError(
        f"{len(invalid_matches)} matches have inconsistent "
        "or missing final scores."
    )

match_df = (
    df.groupby("match_id", as_index=False)
    .agg(
        final_score=("final_score", "first"),
        comp_name=("comp_name", "first"),
        competition_type=("competition_type", "first"),
        season=("season", "first"),
        venue_id=("venue_id", "first"),
        match_date=("match_date", "first")
    )
)

if match_df["match_id"].duplicated().any():
    raise ValueError("Duplicate match records detected.")

print("Match-level records:", len(match_df))
print("IPL matches:", (match_df["competition_type"] == "IPL").sum())
print("T20I matches:", (match_df["competition_type"] == "T20I").sum())


# ==================================================
# 4. SUMMARY FUNCTION
# ==================================================

def calculate_summary(data, group_columns):

    summary = (
        data.groupby(group_columns, dropna=False)
        .agg(
            matches=("match_id", "nunique"),
            mean_score=("final_score", "mean"),
            median_score=("final_score", "median"),
            std_score=("final_score", "std"),
            minimum_score=("final_score", "min"),
            q25=("final_score", lambda x: x.quantile(0.25)),
            q75=("final_score", lambda x: x.quantile(0.75)),
            maximum_score=("final_score", "max")
        )
        .reset_index()
    )

    summary["mean_median_difference"] = (
        summary["mean_score"] - summary["median_score"]
    )

    return summary


# ==================================================
# 5. OVERALL PAR-SCORE BENCHMARK
# ==================================================

print("\n" + "=" * 70)
print("OVERALL FIRST-INNINGS SCORE DISTRIBUTION")
print("=" * 70)

overall = pd.DataFrame([{
    "matches": match_df["match_id"].nunique(),
    "mean_score": match_df["final_score"].mean(),
    "median_score": match_df["final_score"].median(),
    "std_score": match_df["final_score"].std(),
    "minimum_score": match_df["final_score"].min(),
    "q25": match_df["final_score"].quantile(0.25),
    "q75": match_df["final_score"].quantile(0.75),
    "maximum_score": match_df["final_score"].max()
}])

overall["mean_median_difference"] = (
    overall["mean_score"] - overall["median_score"]
)

print(overall.round(2).to_string(index=False))


# ==================================================
# 6. IPL VS T20I
# ==================================================

print("\n" + "=" * 70)
print("COMPETITION-WISE PAR SCORES")
print("=" * 70)

competition_summary = calculate_summary(
    match_df,
    ["competition_type"]
)

print(competition_summary.round(2).to_string(index=False))


# ==================================================
# 7. SEASON-WISE PAR SCORES
# ==================================================

print("\n" + "=" * 70)
print("SEASON-WISE PAR SCORES")
print("=" * 70)

season_summary = calculate_summary(
    match_df,
    ["competition_type", "season"]
)

print(season_summary.round(2).to_string(index=False))


# ==================================================
# 8. VENUE-WISE PAR SCORES
# ==================================================

print("\n" + "=" * 70)
print("VENUE-WISE PAR SCORES")
print("=" * 70)

venue_summary = calculate_summary(
    match_df,
    ["competition_type", "venue_id"]
)

venue_summary = venue_summary.sort_values(
    ["competition_type", "matches"],
    ascending=[True, False]
)

print(
    venue_summary.round(2).to_string(index=False)
)


# ==================================================
# 9. VENUE SAMPLE-SIZE DISTRIBUTION
# ==================================================

print("\n" + "=" * 70)
print("VENUE SAMPLE-SIZE DISTRIBUTION")
print("=" * 70)

venue_counts = (
    match_df.groupby(["competition_type", "venue_id"])
    .agg(matches=("match_id", "nunique"))
    .reset_index()
)

venue_distribution = (
    venue_counts.groupby("competition_type")["matches"]
    .describe(
        percentiles=[0.25, 0.50, 0.75, 0.90]
    )
)

print(venue_distribution.round(2).to_string())


# ==================================================
# 10. SAVE OUTPUTS
# ==================================================

match_df.to_csv(
    OUTPUT / "match_level_first_innings_scores.csv",
    index=False
)

overall.to_csv(
    OUTPUT / "overall_par_score.csv",
    index=False
)

competition_summary.to_csv(
    OUTPUT / "competition_par_scores.csv",
    index=False
)

season_summary.to_csv(
    OUTPUT / "season_par_scores.csv",
    index=False
)

venue_summary.to_csv(
    OUTPUT / "venue_par_scores.csv",
    index=False
)

venue_counts.to_csv(
    OUTPUT / "venue_sample_sizes.csv",
    index=False
)


# ==================================================
# 11. COMPLETION
# ==================================================

print("\n" + "=" * 70)
print("COMPETITIVE PAR-SCORE ANALYSIS COMPLETE")
print("=" * 70)

print("\nOutput folder:")
print(OUTPUT)

print("\nFiles generated:")
print("1. match_level_first_innings_scores.csv")
print("2. overall_par_score.csv")
print("3. competition_par_scores.csv")
print("4. season_par_scores.csv")
print("5. venue_par_scores.csv")
print("6. venue_sample_sizes.csv")
