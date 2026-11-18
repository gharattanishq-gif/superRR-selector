
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data Pack"
OUTPUT = ROOT / "outputs" / "processed"

batting = pd.read_csv(OUTPUT / "player_match_batting.csv")
bowling = pd.read_csv(OUTPUT / "player_match_bowling.csv")
balls = pd.read_csv(DATA / "ball_by_ball.csv")

print("\n1. RECORD COUNTS")
print("Batting:", len(batting))
print("Bowling:", len(bowling))

print("\n2. DUPLICATE PLAYER-MATCH RECORDS")
bat_key = ["match_id", "innings", "striker_id"]
bowl_key = ["match_id", "innings", "bowler_id"]

print("Batting:", batting.duplicated(bat_key).sum())
print("Bowling:", bowling.duplicated(bowl_key).sum())

print("\n3. MATCH COVERAGE")
expected = set(balls.loc[balls["innings"].isin([1, 2]), "match_id"].unique())
print("Expected matches:", len(expected))
print("Batting matches:", batting["match_id"].nunique())
print("Bowling matches:", bowling["match_id"].nunique())
print("Missing batting matches:", len(expected - set(batting["match_id"])))
print("Missing bowling matches:", len(expected - set(bowling["match_id"])))

print("\n4. MISSING MATCH CONTEXT")
context_cols = ["comp_id", "season", "match_date", "venue_id"]
print("Batting:", batting[context_cols].isna().sum().to_dict())
print("Bowling:", bowling[context_cols].isna().sum().to_dict())

print("\n5. INVALID VALUES")
print("Negative batting runs:", (batting["runs"] < 0).sum())
print("Negative balls faced:", (batting["balls_faced"] < 0).sum())
print("Negative bowling runs:", (bowling["runs_conceded"] < 0).sum())
print("Negative legal balls:", (bowling["legal_balls"] < 0).sum())
print("Negative wickets:", (bowling["wickets"] < 0).sum())

print("\n6. UNRESOLVED EXTRAS")
print(balls["extras_type"].value_counts(dropna=False))

print("\nFlagged batting records:",
      (~batting["balls_faced_complete"]).sum())
print("Flagged bowling records:",
      (~bowling["bowling_stats_complete"]).sum())

print("\nVALIDATION COMPLETE")
