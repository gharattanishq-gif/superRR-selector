
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data Pack"
OUTPUT = ROOT / "outputs" / "processed"

states = pd.read_csv(OUTPUT / "first_innings_states.csv", low_memory=False)
matches = pd.read_csv(DATA / "matches.csv")

# Inspect largest remaining-ball values
print("\nLARGEST BALLS REMAINING")

largest = states.nlargest(15, "legal_balls_remaining")

print(largest[[
    "match_id", "comp_name", "season",
    "current_score", "wickets_lost",
    "legal_balls_remaining", "final_score"
]].to_string(index=False))

# Inspect revised overs distribution
print("\nFIRST-INNINGS REVISED OVERS")

print(
    matches["first_inn_revised_overs"]
    .describe()
)

print("\nLARGEST REVISED OVERS")

print(
    matches.nlargest(15, "first_inn_revised_overs")[
        ["match_id", "comp_id", "match_date",
         "first_inn_revised_overs", "revised_target"]
    ].to_string(index=False)
)

# Check the distribution of remaining balls
print("\nREMAINING BALLS DISTRIBUTION")

print(
    states["legal_balls_remaining"]
    .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
)
