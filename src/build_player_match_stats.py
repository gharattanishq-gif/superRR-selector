
import pandas as pd
from pathlib import Path

# ==================================================
# 1. PATHS AND DATA
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data Pack"
OUTPUT = ROOT / "outputs" / "processed"
OUTPUT.mkdir(parents=True, exist_ok=True)

balls = pd.read_csv(DATA / "ball_by_ball.csv")
matches = pd.read_csv(DATA / "matches.csv")
competitions = pd.read_csv(DATA / "competitions.csv")

# Exclude Super Overs
balls = balls[balls["innings"].isin([1, 2])].copy()

# Normalise extras labels
balls["extras_type"] = balls["extras_type"].fillna("none").str.lower()

# ==================================================
# 2. DELIVERY CLASSIFICATION
# ==================================================

balls["is_wide"] = balls["extras_type"].eq("wide")
balls["is_overstep"] = balls["extras_type"].eq("overstep")

# Provisional: Overstep is treated as a no-ball
balls["legal_ball"] = ~(
    balls["is_wide"] | balls["is_overstep"]
)

# Unresolved categories
balls["unresolved_extra"] = balls["extras_type"].isin(
    ["other", "penalty"]
)

# Additional Overstep runs need separate interpretation
balls["unresolved_overstep_extra"] = (
    balls["is_overstep"] & balls["extras_2_runs"].ne(0)
)

# ==================================================
# 3. RUN CALCULATIONS
# ==================================================

# runs represents batter runs
balls["batter_runs"] = balls["runs"]

# Known bowler-conceded runs:
# Batter runs + wide penalty + Overstep penalty
balls["bowler_runs"] = (
    balls["runs"]
    + balls["extras1_runs"].where(
        balls["is_wide"] | balls["is_overstep"], 0
    )
)

# ==================================================
# 4. BATTING STATISTICS
# ==================================================

balls["batter_faced_ball"] = balls["legal_ball"] & ~balls[
    "unresolved_extra"
]

batting = balls.groupby(
    ["match_id", "innings", "batting_team_id",
     "striker_id", "striker_name"]
).agg(
    runs=("batter_runs", "sum"),
    balls_faced=("batter_faced_ball", "sum"),
    fours=("is_four", "sum"),
    sixes=("is_six", "sum"),
    dismissals=("dismissed_player_id",
                lambda x: x.notna().sum()),
    unresolved_deliveries=("unresolved_extra", "sum")
).reset_index()

batting["strike_rate"] = (
    batting["runs"] /
    batting["balls_faced"].replace(0, pd.NA) * 100
)

batting["balls_faced_complete"] = (
    batting["unresolved_deliveries"] == 0
)

# ==================================================
# 5. BOWLING STATISTICS
# ==================================================

# Bowler-credit dismissal types
bowler_wickets = [
    "caught",
    "bowled",
    "lbw",
    "stump",
    "hit wicket"
]

balls["bowler_wicket"] = (
    balls["is_wicket"]
    & balls["dismissal_type"].str.lower().isin(bowler_wickets)
)

bowling = balls.groupby(
    ["match_id", "innings", "bowling_team_id",
     "bowler_id", "bowler_name"]
).agg(
    runs_conceded=("bowler_runs", "sum"),
    legal_balls=("legal_ball", "sum"),
    wickets=("bowler_wicket", "sum"),
    unresolved_deliveries=("unresolved_extra", "sum"),
    unresolved_overstep_runs=("unresolved_overstep_extra", "sum")
).reset_index()

bowling["overs"] = (
    bowling["legal_balls"] // 6
    + (bowling["legal_balls"] % 6) / 10
)

bowling["economy"] = (
    bowling["runs_conceded"] /
    bowling["legal_balls"].replace(0, pd.NA) * 6
)

bowling["bowling_average"] = (
    bowling["runs_conceded"] /
    bowling["wickets"].replace(0, pd.NA)
)

bowling["bowling_strike_rate"] = (
    bowling["legal_balls"] /
    bowling["wickets"].replace(0, pd.NA)
)

bowling["bowling_stats_complete"] = (
    (bowling["unresolved_deliveries"] == 0)
    & (bowling["unresolved_overstep_runs"] == 0)
)

# ==================================================
# 6. MATCH AND COMPETITION CONTEXT
# ==================================================

context = matches.merge(
    competitions[
        ["comp_id", "comp_name", "match_format", "season"]
    ],
    on="comp_id",
    how="left"
)

context = context[[
    "match_id", "comp_id", "comp_name", "match_format",
    "season", "match_date", "match_number", "stage",
    "venue_id", "venue_country", "home_team_id",
    "away_team_id", "knock_out", "revised_target",
    "first_inn_revised_overs", "second_inn_revised_overs"
]]

batting = batting.merge(context, on="match_id", how="left")
bowling = bowling.merge(context, on="match_id", how="left")

# ==================================================
# 7. EXPORT
# ==================================================

batting.to_csv(
    OUTPUT / "player_match_batting.csv", index=False
)

bowling.to_csv(
    OUTPUT / "player_match_bowling.csv", index=False
)

print("\nPLAYER-MATCH STATISTICS GENERATED")
print("-" * 40)
print(f"Matches: {balls['match_id'].nunique()}")
print(f"Batting records: {len(batting)}")
print(f"Bowling records: {len(bowling)}")
print(
    "Batting records with unresolved extras:",
    (batting["balls_faced_complete"] == False).sum()
)
print(
    "Bowling records with unresolved extras:",
    (bowling["bowling_stats_complete"] == False).sum()
)
print(f"\nOutput directory: {OUTPUT}")
