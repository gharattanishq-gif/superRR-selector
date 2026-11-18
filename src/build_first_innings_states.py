
import pandas as pd
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data Pack"
OUTPUT = ROOT / "outputs" / "processed"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Load data
balls = pd.read_csv(DATA / "ball_by_ball.csv")
matches = pd.read_csv(DATA / "matches.csv")
competitions = pd.read_csv(DATA / "competitions.csv")

# Match and competition context
context = matches.merge(
    competitions[["comp_id", "comp_name", "match_format", "season"]],
    on="comp_id",
    how="left"
)

# Revised overs are stored as legal-ball counts
context["max_legal_balls"] = (
    context["first_inn_revised_overs"]
    .fillna(120)
    .astype(int)
)

# Primary competitions: IPL and senior men's T20Is
primary = (
    context["comp_name"].str.contains("IPL", case=False, na=False)
    | context["match_format"].eq("T20I")
)

context = context[primary]

# First innings only; exclude Super Overs
balls = balls[balls["innings"] == 1].copy()
balls = balls.merge(
    context[[
        "match_id", "comp_id", "comp_name", "season",
        "match_date", "venue_id", "max_legal_balls"
    ]],
    on="match_id",
    how="inner"
)

# Delivery classification
balls["extras_type"] = (
    balls["extras_type"].fillna("none").astype(str).str.lower()
)

# Provisional: Overstep treated as a no-ball
balls["legal_ball"] = ~balls["extras_type"].isin(
    ["wide", "overstep"]
)

# Exclude innings with unresolved extras
balls["unresolved"] = balls["extras_type"].isin(
    ["other", "penalty"]
)

# Total team runs recorded on each delivery
balls["delivery_runs"] = (
    balls["runs"]
    + balls["extras1_runs"].fillna(0)
    + balls["extras_2_runs"].fillna(0)
)

# Sort deliveries chronologically within each innings
balls = balls.sort_values(
    ["match_id", "overid", "over_ball_num"]
).reset_index(drop=True)

group = balls.groupby(["match_id", "innings"], sort=False)

# Check innings quality
balls["innings_has_unresolved"] = group["unresolved"].transform("max")
balls["innings_legal_balls"] = group["legal_ball"].transform("sum")

balls["valid_innings"] = (
    ~balls["innings_has_unresolved"]
    & (balls["innings_legal_balls"] <= balls["max_legal_balls"])
)

excluded = balls.loc[~balls["valid_innings"], "match_id"].nunique()
balls = balls[balls["valid_innings"]].copy()

# Recalculate group after filtering
group = balls.groupby(["match_id", "innings"], sort=False)

# Match state BEFORE each delivery
balls["current_score"] = (
    group["delivery_runs"].cumsum() - balls["delivery_runs"]
)

balls["wickets_lost"] = (
    group["is_wicket"].cumsum() - balls["is_wicket"].astype(int)
)

balls["legal_balls_bowled"] = (
    group["legal_ball"].cumsum() - balls["legal_ball"].astype(int)
)

balls["legal_balls_remaining"] = (
    balls["max_legal_balls"] - balls["legal_balls_bowled"]
)

# Final innings score: prediction target
balls["final_score"] = group["delivery_runs"].transform("sum")

# Prepare modelling dataset
features = [
    "match_id", "comp_id", "comp_name", "season",
    "match_date", "venue_id", "current_score",
    "wickets_lost", "legal_balls_remaining", "final_score"
]

model_data = balls[features].copy()
model_data["match_date"] = pd.to_datetime(model_data["match_date"])

# Export
model_data.to_csv(
    OUTPUT / "first_innings_states.csv",
    index=False
)

print("\nFIRST-INNINGS MODELLING DATA GENERATED")
print("-" * 45)
print("Modelling rows:", len(model_data))
print("Matches:", model_data["match_id"].nunique())
print("Competitions:", model_data["comp_name"].nunique())
print("Excluded matches:", excluded)
print("Missing values:", model_data.isna().sum().sum())
print("Output: first_innings_states.csv")
