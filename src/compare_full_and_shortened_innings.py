
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "outputs" / "processed" / "first_innings_states.csv"
OUTPUT_DIR = ROOT / "outputs" / "analysis" / "par_scores"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE, low_memory=False)

required_columns = [
    "match_id",
    "final_score",
    "legal_balls_remaining",
    "comp_name",
    "season",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# --------------------------------------------------
# CREATE MATCH-LEVEL DATA
# --------------------------------------------------

# Each match should have one final first-innings score.
# Maximum legal balls remaining represents the scheduled
# first-innings length at the start of the innings.

matches = (
    df.groupby("match_id")
    .agg(
        final_score=("final_score", "first"),
        score_variations=("final_score", "nunique"),
        initial_legal_balls=("legal_balls_remaining", "max"),
        comp_name=("comp_name", "first"),
        season=("season", "first"),
    )
    .reset_index()
)

if (matches["score_variations"] != 1).any():
    raise ValueError("Some matches have inconsistent final scores.")

matches["competition_type"] = matches["comp_name"].apply(
    lambda x: "IPL" if "IPL" in str(x).upper() else "T20I"
)

matches["innings_type"] = matches["initial_legal_balls"].apply(
    lambda x: "Full 120-ball" if x == 120 else "Shortened"
)

if (matches["initial_legal_balls"] <= 0).any():
    raise ValueError("Invalid initial legal-ball limit detected.")

if (matches["initial_legal_balls"] > 120).any():
    raise ValueError("Unexpected first-innings limit above 120 balls.")


# --------------------------------------------------
# SUMMARY FUNCTION
# --------------------------------------------------

def summarise(data, group_columns):
    return (
        data.groupby(group_columns, dropna=False)
        .agg(
            matches=("match_id", "nunique"),
            mean_score=("final_score", "mean"),
            median_score=("final_score", "median"),
            std_score=("final_score", "std"),
            minimum=("final_score", "min"),
            q1=("final_score", lambda x: x.quantile(0.25)),
            q3=("final_score", lambda x: x.quantile(0.75)),
            maximum=("final_score", "max"),
            mean_initial_balls=("initial_legal_balls", "mean"),
        )
        .reset_index()
        .round(2)
    )


# --------------------------------------------------
# 1. OVERALL COMPARISON
# --------------------------------------------------

overall = summarise(matches, ["innings_type"])

overall.to_csv(
    OUTPUT_DIR / "full_vs_shortened_overall.csv",
    index=False,
)


# --------------------------------------------------
# 2. COMPETITION-WISE COMPARISON
# --------------------------------------------------

competition = summarise(
    matches,
    ["competition_type", "innings_type"],
)

competition.to_csv(
    OUTPUT_DIR / "full_vs_shortened_competition.csv",
    index=False,
)


# --------------------------------------------------
# 3. FULL-OVERS COMPETITION BENCHMARKS
# --------------------------------------------------

full_overs = matches[matches["initial_legal_balls"] == 120].copy()

full_benchmarks = summarise(
    full_overs,
    ["competition_type"],
)

full_benchmarks.to_csv(
    OUTPUT_DIR / "full_overs_competition_benchmarks.csv",
    index=False,
)


# --------------------------------------------------
# 4. SEASON-WISE FULL-OVERS COMPARISON
# --------------------------------------------------

seasonal = summarise(
    full_overs,
    ["competition_type", "season"],
)

seasonal.to_csv(
    OUTPUT_DIR / "full_overs_season_benchmarks.csv",
    index=False,
)


# --------------------------------------------------
# 5. SHORTENED INNINGS DETAILS
# --------------------------------------------------

shortened = matches[matches["initial_legal_balls"] < 120].copy()

shortened_summary = summarise(
    shortened,
    ["competition_type", "initial_legal_balls"],
)

shortened_summary.to_csv(
    OUTPUT_DIR / "shortened_innings_details.csv",
    index=False,
)


# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("FULL VS SHORTENED FIRST-INNINGS ANALYSIS")
print("=" * 65)

print(f"\nTotal matches analysed: {len(matches)}")
print(f"Full 120-ball innings: {len(full_overs)}")
print(f"Shortened innings: {len(shortened)}")

print("\n--- OVERALL COMPARISON ---")
print(overall.to_string(index=False))

print("\n--- COMPETITION-WISE COMPARISON ---")
print(competition.to_string(index=False))

print("\n--- FULL-OVERS COMPETITION BENCHMARKS ---")
print(full_benchmarks.to_string(index=False))

print("\n--- FULL-OVERS SEASON BENCHMARKS ---")
print(seasonal.to_string(index=False))

print("\n--- SHORTENED INNINGS DETAILS ---")
print(shortened_summary.to_string(index=False))

print("\n" + "=" * 65)
print(f"Analysis saved to: {OUTPUT_DIR}")
print("=" * 65)
