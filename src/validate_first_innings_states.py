
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data Pack"
OUTPUT = ROOT / "outputs" / "processed"

states = pd.read_csv(OUTPUT / "first_innings_states.csv")
balls = pd.read_csv(DATA / "ball_by_ball.csv")

print("\n1. DATASET SUMMARY")
print("Rows:", len(states))
print("Matches:", states["match_id"].nunique())

print("\n2. STATE VALIDATION")
print("Negative scores:", (states["current_score"] < 0).sum())
print("Negative wickets:", (states["wickets_lost"] < 0).sum())
print("Negative balls remaining:",
      (states["legal_balls_remaining"] < 0).sum())

print("\n3. FINAL SCORE CONSISTENCY")
last_states = states.sort_values(
    ["match_id", "legal_balls_remaining"]
).groupby("match_id").tail(1)

print("Matches checked:", len(last_states))
print("Final score mismatches:",
      (last_states["current_score"] > last_states["final_score"]).sum())

print("\n4. MATCH-LEVEL DISTRIBUTIONS")
print(
    states.groupby("match_id").agg(
        final_score=("final_score", "first"),
        max_score=("current_score", "max"),
        min_balls_remaining=("legal_balls_remaining", "min")
    ).describe().round(2)
)

print("\n5. SOURCE EXCLUSION CANDIDATES")
first = balls[balls["innings"] == 1].copy()
first["extras_type"] = first["extras_type"].fillna("none").astype(str).str.lower()

print("Matches with other/penalty:",
      first.loc[first["extras_type"].isin(["other", "penalty"]),
                "match_id"].nunique())

print("Matches with Overstep:",
      first.loc[first["extras_type"] == "overstep",
                "match_id"].nunique())

print("\nVALIDATION COMPLETE")
