
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
balls = pd.read_csv(ROOT / "Data Pack" / "ball_by_ball.csv")

print("\nOTHER EXTRAS: ALL COMBINATIONS")
print(
    balls[balls["extras_type"] == "other"]
    .groupby(["runs", "extras1_runs", "extras_2_runs"])
    .size()
    .sort_values(ascending=False)
)

print("\nOVERSTEP: ALL COMBINATIONS")
print(
    balls[balls["extras_type"] == "Overstep"]
    .groupby(["runs", "extras1_runs", "extras_2_runs"])
    .size()
    .sort_values(ascending=False)
)

for category in ["Overstep", "other"]:
    subset = balls[balls["extras_type"] == category]

    print(f"\n{category}")
    print(subset["dismissal_type"].value_counts(dropna=False))
    print("Wicket deliveries:", subset["is_wicket"].sum())