
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

OVERALL_DIR = ROOT / "outputs" / "analysis" / "overall_indices"
PRESSURE_DIR = ROOT / "outputs" / "analysis" / "final_pressure_indices"
OUTPUT_DIR = ROOT / "outputs" / "analysis" / "combined_player_scores"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def normalise_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def find_column(df, candidates, description):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not find {description}. "
        f"Available columns: {list(df.columns)}"
    )


def prepare_overall(path, index_type):
    df = normalise_columns(read_csv(path))

    player_col = find_column(
        df,
        ["player_id", "entity_id", "striker_id", "bowler_id"],
        "player identifier",
    )

    name_col = find_column(
        df,
        ["player_name", "entity_name", "striker_name", "bowler_name", "name"],
        "player name",
    )

    population_col = find_column(
        df, ["population", "competition"], "population"
    )

    period_col = find_column(
        df,
        ["period", "season", "evaluation_period"],
        "evaluation period",
    )

    score_col = find_column(
        df,
        [
            "overall_index",
            "overall_score",
            "index_score",
            "overall_index_percentile",
            "score",
        ],
        "overall index score",
    )

    result = df[
        [player_col, name_col, population_col, period_col, score_col]
    ].copy()

    result.columns = [
        "player_id",
        "player_name",
        "population",
        "period",
        "overall_index",
    ]

    result["player_id"] = result["player_id"].astype(str)
    result["period"] = result["period"].astype(str)

    result["overall_index"] = pd.to_numeric(
        result["overall_index"], errors="coerce"
    )

    result["index_type"] = index_type

    return result

def prepare_pressure(path, index_type):
    df = normalise_columns(read_csv(path))

    player_col = find_column(
        df, ["player_id"], "player_id"
    )
    name_col = find_column(
        df, ["player_name", "name"], "player name"
    )
    population_col = find_column(
        df, ["population", "competition"], "population"
    )
    period_col = find_column(
        df, ["period", "season", "evaluation_period"], "period"
    )
    score_col = find_column(
    df,
    [
        "pressure_index",
        "pressure_performance_index",
        "ppi_score",
        "ppi",
    ],
    "pressure index score",
    )

    result = df[
        [player_col, name_col, population_col, period_col, score_col]
    ].copy()

    result.columns = [
        "player_id", "player_name", "population", "period", "pressure_index"
    ]

    result["player_id"] = result["player_id"].astype(str)
    result["period"] = result["period"].astype(str)
    result["pressure_index"] = pd.to_numeric(
        result["pressure_index"], errors="coerce"
    )

    result["index_type"] = index_type
    return result


def build_competition_scores(index_type):
    overall_file = (
        OVERALL_DIR / f"player_overall_{index_type}_indices.csv"
    )
    pressure_file = (
        PRESSURE_DIR / f"player_{index_type}_pressure_index.csv"
    )

    overall = prepare_overall(overall_file, index_type)
    pressure = prepare_pressure(pressure_file, index_type)

    keys = ["player_id", "population", "period"]

    if overall.duplicated(keys).any():
        raise ValueError(
            f"Duplicate overall-index keys found for {index_type}"
        )

    if pressure.duplicated(keys).any():
        raise ValueError(
            f"Duplicate pressure-index keys found for {index_type}"
        )

    merged = overall.merge(
        pressure,
        on=keys,
        how="left",
        suffixes=("_overall", "_pressure"),
        validate="one_to_one",
    )

    merged["player_name"] = (
        merged["player_name_overall"]
        .fillna(merged["player_name_pressure"])
    )

    merged["overall_available"] = (
        merged["overall_index"].notna()
    )
    merged["pressure_available"] = (
        merged["pressure_index"].notna()
    )

    # Combined score requires both components.
    merged["combined_score"] = np.where(
        merged["overall_available"] & merged["pressure_available"],
        0.35 * merged["overall_index"]
        + 0.65 * merged["pressure_index"],
        np.nan,
    )

    merged["combined_rank"] = (
        merged.groupby(["population", "period"])["combined_score"]
        .rank(method="min", ascending=False)
    )

    merged["index_type"] = index_type

    return merged


def build_cross_competition_scores(competition_scores, index_type):
    df = competition_scores.copy()

    # IPL rows are season-specific.
    ipl = df[
        df["population"].astype(str).str.contains(
            "IPL", case=False, na=False
        )
        & df["combined_score"].notna()
    ].copy()

    # Actual source label is "Senior men's T20I", not simply "T20I".
    t20i = df[
        df["population"].astype(str).str.contains(
            "T20I", case=False, na=False
        )
        & df["combined_score"].notna()
    ].copy()

    if ipl.empty or t20i.empty:
        return pd.DataFrame()

    t20i_any = (
        t20i.sort_values("period")
        .drop_duplicates("player_id", keep="last")
        [["player_id", "player_name", "combined_score"]]
        .rename(
            columns={
                "player_name": "t20i_player_name",
                "combined_score": "t20i_combined_score",
            }
        )
    )

    paired = ipl.merge(
        t20i_any,
        on="player_id",
        how="inner",
        validate="many_to_one",
    )

    paired["cross_competition_score"] = (
        0.65 * paired["combined_score"]
        + 0.35 * paired["t20i_combined_score"]
    )

    paired["cross_competition_rank"] = (
        paired.groupby("period")["cross_competition_score"]
        .rank(method="min", ascending=False)
    )

    paired["index_type"] = index_type

    return paired


def main():
    batting = build_competition_scores("batting")
    bowling = build_competition_scores("bowling")

    batting.to_csv(
        OUTPUT_DIR / "combined_batting_scores_by_competition.csv",
        index=False,
    )
    bowling.to_csv(
        OUTPUT_DIR / "combined_bowling_scores_by_competition.csv",
        index=False,
    )

    all_scores = pd.concat(
        [batting, bowling],
        ignore_index=True,
    )

    ranking_columns = [
        "index_type",
        "player_id",
        "player_name",
        "population",
        "period",
        "overall_index",
        "pressure_index",
        "combined_score",
        "combined_rank",
        "overall_available",
        "pressure_available",
    ]

    all_scores[ranking_columns].to_csv(
        OUTPUT_DIR / "combined_player_rankings.csv",
        index=False,
    )

    cross_batting = build_cross_competition_scores(
        batting, "batting"
    )
    cross_bowling = build_cross_competition_scores(
        bowling, "bowling"
    )

    cross = pd.concat(
        [cross_batting, cross_bowling],
        ignore_index=True,
    )

    cross.to_csv(
        OUTPUT_DIR / "cross_competition_rankings.csv",
        index=False,
    )

    validation = []

    for index_type, df in [
        ("batting", batting),
        ("bowling", bowling),
    ]:
        for (population, period), group in df.groupby(
            ["population", "period"]
        ):
            validation.append(
                {
                    "index_type": index_type,
                    "population": population,
                    "period": period,
                    "eligible_players": len(group),
                    "overall_available": int(
                        group["overall_available"].sum()
                    ),
                    "pressure_available": int(
                        group["pressure_available"].sum()
                    ),
                    "combined_available": int(
                        group["combined_score"].notna().sum()
                    ),
                    "duplicate_player_keys": int(
                        group.duplicated(
                            ["player_id", "population", "period"]
                        ).sum()
                    ),
                    "out_of_range_scores": int(
                        (
                            group["combined_score"].notna()
                            & ~group["combined_score"].between(0, 100)
                        ).sum()
                    ),
                }
            )

    validation_df = pd.DataFrame(validation)
    validation_df.to_csv(
        OUTPUT_DIR / "combined_score_validation.csv",
        index=False,
    )

    cross_coverage = []

    if not cross.empty:
        for (index_type, period), group in cross.groupby(
            ["index_type", "period"]
        ):
            cross_coverage.append(
                {
                    "index_type": index_type,
                    "period": period,
                    "players_with_both_competition_scores": (
                        group["player_id"].nunique()
                    ),
                }
            )

    pd.DataFrame(cross_coverage).to_csv(
        OUTPUT_DIR / "cross_competition_coverage.csv",
        index=False,
    )

    print("\nCOMBINED PLAYER SCORES BUILT")
    print(f"Output directory: {OUTPUT_DIR}")

    print("\nVALIDATION")
    print(validation_df.to_string(index=False))

    print("\nCROSS-COMPETITION COVERAGE")
    if cross_coverage:
        print(pd.DataFrame(cross_coverage).to_string(index=False))
    else:
        print("No players have complete evidence in both competitions.")

    print("\nTOP 10: COMPETITION-SPECIFIC COMBINED SCORES")

    for index_type, df in [
        ("BATTING", batting),
        ("BOWLING", bowling),
    ]:
        print(f"\n{index_type}")

        for (population, period), group in df.groupby(
            ["population", "period"]
        ):
            top = (
                group.dropna(subset=["combined_score"])
                .sort_values("combined_score", ascending=False)
                .head(10)
            )

            print(f"\n{population} | {period}")
            if top.empty:
                print("No complete combined scores.")
            else:
                print(
                    top[
                        [
                            "player_name",
                            "overall_index",
                            "pressure_index",
                            "combined_score",
                        ]
                    ].to_string(index=False)
                )

    print("\nTOP 10: CROSS-COMPETITION SCORES")

    for index_type, df in [
        ("BATTING", cross_batting),
        ("BOWLING", cross_bowling),
    ]:
        print(f"\n{index_type}")

        if df.empty:
            print("No paired competition scores.")
            continue

        for period, group in df.groupby("period"):
            top = group.sort_values(
                "cross_competition_score",
                ascending=False,
            ).head(10)

            print(f"\nIPL {period} + T20I")
            print(
                top[
                    [
                        "player_name",
                        "combined_score",
                        "t20i_combined_score",
                        "cross_competition_score",
                    ]
                ].to_string(index=False)
            )


if __name__ == "__main__":
    main()
