
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data Pack"

files = [
    "ball_by_ball.csv",
    "matches.csv",
    "competitions.csv",
    "playing_eleven.csv",
    "competition_team_player_mapping.csv",
    "teams.csv"
]

for file in files:
    df = pd.read_csv(DATA / file, nrows=5)

    print(f"\n{'=' * 65}")
    print(file)
    print("=" * 65)

    for col, dtype in df.dtypes.items():
        print(f"{col:<35} {dtype}")
