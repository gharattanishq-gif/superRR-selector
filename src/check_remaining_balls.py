
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs" / "processed"

df = pd.read_csv(
    OUTPUT / "first_innings_states.csv",
    low_memory=False
)

print("\nMATCHES WITH HIGHEST INITIAL BALLS REMAINING")

first_states = (
    df.sort_values(["match_id", "legal_balls_remaining"],
                   ascending=[True, False])
    .groupby("match_id")
    .head(1)
)

print(
    first_states.nlargest(15, "legal_balls_remaining")[
        ["match_id", "comp_name", "season",
         "legal_balls_remaining", "final_score"]
    ].to_string(index=False)
)

print("\nINITIAL BALLS REMAINING DISTRIBUTION")
print(
    first_states["legal_balls_remaining"]
    .describe(percentiles=[.25, .5, .75, .9, .95, .99])
)

print("\nINNINGS WITH LESS THAN 120 INITIAL BALLS")
print(
    (first_states["legal_balls_remaining"] < 120).sum(),
    "out of", len(first_states)
)
