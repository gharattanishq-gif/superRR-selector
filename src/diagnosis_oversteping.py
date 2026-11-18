
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data Pack"

balls = pd.read_csv(DATA / "ball_by_ball.csv")
matches = pd.read_csv(DATA / "matches.csv")
competitions = pd.read_csv(DATA / "competitions.csv")

# Extras breakdown
print("\nEXTRAS RUN PATTERNS")
print(
    balls.groupby("extras_type", dropna=False)[
        ["extras1_runs", "extras_2_runs", "runs"]
    ].agg(["sum", "min", "max"])
)

print("\nEXTRAS COMBINATIONS")
print(
    balls.groupby(
        ["extras_type", "extras1_runs", "extras_2_runs"],
        dropna=False
    ).size().sort_values(ascending=False).head(20)
)

# Investigate bowlers exceeding 24 legal balls
balls["legal_ball"] = ~balls["extras_type"].isin(
    ["wide", "noballs", "Overstep"]
)
balls["legal_ball"] = balls["legal_ball"].fillna(True)

long_spells = balls[balls["innings"].isin([1, 2])].groupby(
    ["match_id", "innings", "bowler_id", "bowler_name"]
)["legal_ball"].sum().reset_index()

long_spells = long_spells[long_spells["legal_ball"] > 24]

details = long_spells.merge(matches, on="match_id").merge(
    competitions[["comp_id", "comp_name", "match_format", "season"]],
    on="comp_id"
)

print("\nBOWLERS ABOVE 24 LEGAL BALLS")
print(details[[
    "match_id", "innings", "bowler_name", "legal_ball",
    "comp_name", "match_format", "season",
    "first_inn_revised_overs", "second_inn_revised_overs"
]].to_string(index=False))
