
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs" / "processed"

batting = pd.read_csv(OUTPUT / "player_match_batting.csv")
bowling = pd.read_csv(OUTPUT / "player_match_bowling.csv")

print("\nBATTING SUMMARY")
print(batting[["runs", "balls_faced", "fours", "sixes", "strike_rate"]].describe().round(2))

print("\nBOWLING SUMMARY")
print(bowling[["runs_conceded", "legal_balls", "wickets", "economy"]].describe().round(2))

print("\nMISSING VALUES")
print("Batting:", batting.isna().sum().sum())
print("Bowling:", bowling.isna().sum().sum())

print("\nINVALID VALUES")
print("Negative batting runs:", (batting["runs"] < 0).sum())
print("Negative bowling runs:", (bowling["runs_conceded"] < 0).sum())
print("Negative legal balls:", (bowling["legal_balls"] < 0).sum())
