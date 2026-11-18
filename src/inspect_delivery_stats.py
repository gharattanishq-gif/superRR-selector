
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data Pack"

balls = pd.read_csv(DATA / "ball_by_ball.csv")

print("\nEXTRAS TYPES")
print(balls["extras_type"].value_counts(dropna=False))

print("\nEXTRA RUN DISTRIBUTION")
print(balls.groupby("extras_type", dropna=False)[
    ["extras1_runs", "extras_2_runs", "runs"]
].sum())

print("\nDISMISSAL TYPES")
print(balls.loc[balls["is_wicket"], "dismissal_type"].value_counts(dropna=False))

print("\nLEGAL BALLS PER BOWLER-MATCH")
legal = ~balls["extras_type"].isin(["wide", "noballs"])
legal = legal.fillna(True)

bowling = balls.assign(legal_ball=legal).groupby(
    ["match_id", "bowler_id"]
)["legal_ball"].sum()

print(bowling.describe())

print("\nBOWLING RECORDS ABOVE 24 LEGAL BALLS")
print(bowling[bowling > 24].sort_values(ascending=False).head(15))
