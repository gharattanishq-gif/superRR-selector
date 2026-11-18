
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
balls = pd.read_csv(ROOT / "Data Pack" / "ball_by_ball.csv")

for category in ["wide", "byes", "legbyes", "Overstep", "other", "Penalty"]:
    print(f"\n--- {category} ---")

    cols = [
        "match_id", "innings", "overid", "over_ball_num",
        "striker_name", "bowler_name", "runs",
        "extras_type", "extras1_runs", "extras_2_runs"
    ]

    print(
        balls.loc[balls["extras_type"] == category, cols]
        .head(5)
        .to_string(index=False)
    )
