from pathlib import Path
import numpy as np
import pandas as pd

# SupeRR Selector Season 2 — Round 1
# Final metric-quality builder with metric-specific completeness
# and reconstructed batter balls faced.
#
# All bowling performance metrics are reconstructed from the same
# delivery-level population. Existing player-match bowling totals
# are used only as reconciliation references.

def find_root():
    here = Path(__file__).resolve()
    for candidate in [Path.cwd(), *here.parents]:
        if (candidate / "Data Pack").exists() and (candidate / "outputs" / "processed").exists():
            return candidate
    raise FileNotFoundError(
        "Project root not found. Put this script in SupeRR/src and run it from the project."
    )


ROOT = find_root()
DATA = ROOT / "Data Pack"
PROC = ROOT / "outputs" / "processed"
OUT = ROOT / "outputs" / "analysis" / "player_metrics"
OUT.mkdir(parents=True, exist_ok=True)

PATHS = {
    "balls": DATA / "ball_by_ball.csv",
    "matches": DATA / "matches.csv",
    "competitions": DATA / "competitions.csv",
    "batting": PROC / "player_match_batting.csv",
    "bowling": PROC / "player_match_bowling.csv",
}

for label, path in PATHS.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} input: {path}")

BOWLER_CREDITED_DISMISSALS = {
    "caught", "bowled", "lbw", "stump", "hit wicket"
}

KNOWN_EXTRA_TYPES = {"", "wide", "overstep", "byes", "legbyes"}


def require(df, cols, label):
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f"{label}: missing columns: {missing}")


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def clean(series):
    return series.fillna("").astype(str).str.strip().str.lower()


def classify_population(df):
    d = df.copy()
    d["match_format"] = clean(d["match_format"]).str.upper()
    d["comp_name"] = d["comp_name"].fillna("").astype(str).str.strip()
    d["match_date"] = pd.to_datetime(d["match_date"], errors="coerce")

    d = d[
        d["match_date"].between("2022-01-01", "2026-12-31")
    ].copy()

    ipl = (
        d["match_format"].eq("T20")
        & d["comp_name"].str.fullmatch(
            r"IPL\s+\d{4}", case=False, na=False
        )
    )

    t20i = d["match_format"].eq("T20I")
    d = d[ipl | t20i].copy()

    d["population"] = np.where(
        ipl.loc[d.index], "IPL", "Senior men's T20I"
    )

    d["evaluation_period"] = np.where(
        d["population"].eq("IPL"),
        d["season"].astype(str),
        "2022-2026 combined"
    )

    return d


def add_population_from_matches(df, match_meta):
    d = df.merge(
        match_meta[
            [
                "match_id",
                "comp_id_y",
                "comp_name_meta",
                "match_format_meta",
                "season_meta"
            ]
        ],
        on="match_id",
        how="left",
        validate="many_to_one"
    )

    if d["comp_id"].isna().any():
        raise ValueError(
            "Some player-match records do not map to matches/competitions."
        )

    # Match-level metadata is authoritative.
    fmt = clean(d["match_format"]).str.upper()
    meta_fmt = clean(d["match_format_meta"]).str.upper()

    mismatch = (
        fmt.ne(meta_fmt)
        | d["comp_id"].ne(d["comp_id_y"])
    )

    if mismatch.any():
        raise ValueError(
            f"{int(mismatch.sum())} player-match rows have inconsistent competition metadata."
        )

    d["comp_name"] = d["comp_name_meta"]
    d["match_format"] = d["match_format_meta"]
    d["season"] = d["season_meta"]

    d = d.drop(
        columns=[
            "comp_id",
            "comp_id_y",
            "comp_name_meta",
            "match_format_meta",
            "season_meta"
        ]
    )

    return classify_population(d)


def main():

    bat = pd.read_csv(PATHS["batting"], low_memory=False)
    bowl_ref = pd.read_csv(PATHS["bowling"], low_memory=False)
    balls = pd.read_csv(PATHS["balls"], low_memory=False)
    matches = pd.read_csv(PATHS["matches"], low_memory=False)
    comps = pd.read_csv(PATHS["competitions"], low_memory=False)

    require(matches, ["match_id", "comp_id"], "matches")

    require(
        comps,
        ["comp_id", "comp_name", "match_format", "season"],
        "competitions"
    )

    require(
        bat,
        [
            "match_id", "innings", "striker_id", "striker_name",
            "runs", "balls_faced", "fours", "sixes", "dismissals",
            "unresolved_deliveries", "comp_id", "comp_name",
            "match_format", "season", "match_date"
        ],
        "player_match_batting"
    )

    require(
        bowl_ref,
        [
            "match_id", "innings", "bowler_id", "bowler_name",
            "runs_conceded", "legal_balls", "wickets",
            "unresolved_deliveries", "comp_id", "comp_name",
            "match_format", "season", "match_date"
        ],
        "player_match_bowling"
    )

    require(
        balls,
        [
            "match_id", "innings", "overid", "over_ball_num",
            "striker_id", "bowler_id", "striker_name", "bowler_name",
            "runs", "is_four", "is_six", "is_wicket",
            "dismissal_type", "extras_type",
            "extras1_runs", "extras_2_runs"
        ],
        "ball_by_ball"
    )

    match_meta = matches[
        ["match_id", "comp_id"]
    ].merge(
        comps[
            ["comp_id", "comp_name", "match_format", "season"]
        ],
        on="comp_id",
        how="left",
        validate="many_to_one"
    ).rename(
        columns={
            "comp_name": "comp_name_meta",
            "match_format": "match_format_meta",
            "season": "season_meta",
            "comp_id": "comp_id_y"
        }
    )

    if match_meta["comp_name_meta"].isna().any():
        raise ValueError(
            "At least one match has no competition metadata."
        )

    # Match-level population metadata is authoritative.
    bat = add_population_from_matches(bat, match_meta)
    bowl_ref = add_population_from_matches(bowl_ref, match_meta)

    # Exclude Super Overs.
    bat["innings"] = numeric(bat["innings"])
    bowl_ref["innings"] = numeric(bowl_ref["innings"])

    bat = bat[bat["innings"].isin([1, 2])].copy()
    bowl_ref = bowl_ref[bowl_ref["innings"].isin([1, 2])].copy()

    batter_key = ["match_id", "innings", "striker_id"]
    bowler_key = ["match_id", "innings", "bowler_id"]

    if bat.duplicated(batter_key).any():
        raise ValueError(
            "Duplicate batter match-innings keys in player_match_batting."
        )

    if bowl_ref.duplicated(bowler_key).any():
        raise ValueError(
            "Duplicate bowler match-innings keys in player_match_bowling."
        )

    # ---------------------------------------------------------
    # TEAM INNINGS TOTALS
    # ---------------------------------------------------------

    balls["match_id"] = numeric(balls["match_id"])
    balls["innings"] = numeric(balls["innings"])

    balls = balls[balls["innings"].isin([1, 2])].copy()

    for c in ["runs", "extras1_runs", "extras_2_runs"]:
        balls[c] = numeric(balls[c]).fillna(0)

    balls["team_runs_delivery"] = (
        balls["runs"]
        + balls["extras1_runs"]
        + balls["extras_2_runs"]
    )

    innings_totals = balls.groupby(
        ["match_id", "innings"],
        as_index=False
    ).agg(
        team_innings_runs=("team_runs_delivery", "sum")
    )

    # ---------------------------------------------------------
    # BATTER METRICS
    # ---------------------------------------------------------

    balls["extras_clean"] = clean(balls["extras_type"])

    b_e1 = numeric(balls["extras1_runs"]).fillna(0)
    b_e2 = numeric(balls["extras_2_runs"]).fillna(0)

    b_missing_type_extras = (
        balls["extras_clean"].eq("")
        & (b_e1.ne(0) | b_e2.ne(0))
    )

    b_ambiguous_type = balls["extras_clean"].isin(
        ["other", "penalty"]
    )

    b_unknown_type = ~balls["extras_clean"].isin(
        KNOWN_EXTRA_TYPES
    )

    balls["batting_delivery_ambiguous"] = (
        b_missing_type_extras
        | b_ambiguous_type
        | b_unknown_type
        | b_e2.ne(0)
    )

    balls["batter_faced_ball"] = ~balls["extras_clean"].isin(
        ["wide", "overstep"]
    )

    balls.loc[
        balls["batting_delivery_ambiguous"],
        "batter_faced_ball"
    ] = False

    balls["striker_id"] = numeric(balls["striker_id"])
    balls["match_id"] = numeric(balls["match_id"])
    balls["innings"] = numeric(balls["innings"])

    batter_delivery = balls[
        balls["innings"].isin([1, 2])
        & balls["striker_id"].notna()
    ].groupby(
        ["match_id", "innings", "striker_id"],
        as_index=False
    ).agg(
        reconstructed_balls_faced=("batter_faced_ball", "sum"),
        ambiguous_batting_deliveries=(
            "batting_delivery_ambiguous", "sum"
        ),
        batter_delivery_rows=("striker_id", "size")
    )

    bat = bat.merge(
        innings_totals,
        on=["match_id", "innings"],
        how="left",
        validate="many_to_one"
    )

    if bat["team_innings_runs"].isna().any():
        raise ValueError(
            "Missing team innings totals for one or more batter records."
        )

    # Preserve source missingness before filling values.
    for c in ["runs", "balls_faced", "fours", "sixes", "dismissals"]:
        bat[f"{c}_source_missing"] = numeric(bat[c]).isna()

    for c in [
        "runs", "balls_faced", "fours", "sixes",
        "dismissals", "unresolved_deliveries"
    ]:
        bat[c] = numeric(bat[c]).fillna(0)

    bat["striker_id"] = numeric(bat["striker_id"])

    bat = bat.merge(
        batter_delivery,
        on=["match_id", "innings", "striker_id"],
        how="left",
        validate="one_to_one"
    )

    bat[
        [
            "reconstructed_balls_faced",
            "ambiguous_batting_deliveries",
            "batter_delivery_rows"
        ]
    ] = bat[
        [
            "reconstructed_balls_faced",
            "ambiguous_batting_deliveries",
            "batter_delivery_rows"
        ]
    ].fillna(0)

    bat["balls_faced_difference"] = (
        bat["reconstructed_balls_faced"]
        - bat["balls_faced"]
    )

    if "balls_faced_complete" in bat:
        source_complete = bat["balls_faced_complete"]

        if source_complete.dtype != bool:
            source_complete = clean(source_complete).isin(
                ["1", "true", "t", "yes", "y"]
            )
    else:
        source_complete = bat["unresolved_deliveries"].eq(0)

    bat["balls_faced_complete"] = (
        source_complete
        & bat["ambiguous_batting_deliveries"].eq(0)
        & bat["balls_faced_difference"].eq(0)
    )

    bat["boundary_runs"] = (
        4 * bat["fours"] + 6 * bat["sixes"]
    )

    bg = [
        "population",
        "evaluation_period",
        "striker_id",
        "striker_name"
    ]

    batting = bat.groupby(
        bg,
        dropna=False
    ).agg(
        matches_played=("match_id", "nunique"),
        innings_batted=("innings", "count"),
        runs=("runs", "sum"),
        source_balls_faced=("balls_faced", "sum"),
        reconstructed_balls_faced=("reconstructed_balls_faced", "sum"),
        fours=("fours", "sum"),
        sixes=("sixes", "sum"),
        dismissals=("dismissals", "sum"),
        boundary_runs=("boundary_runs", "sum"),
        team_runs_in_batted_innings=("team_innings_runs", "sum"),
        unresolved_deliveries=("unresolved_deliveries", "sum"),
        ambiguous_batting_deliveries=(
            "ambiguous_batting_deliveries", "sum"
        ),
        balls_faced_mismatch_innings=(
            "balls_faced_difference",
            lambda x: int(x.ne(0).sum())
        ),
        incomplete_balls_faced_innings=(
            "balls_faced_complete",
            lambda x: int((~x).sum())
        ),
        first_match=("match_date", "min"),
        last_match=("match_date", "max")
    ).reset_index()

    # Strike rate: only innings with fully reconciled balls faced.
    # Both runs and balls faced use the same complete innings.

    bat["complete_bf_runs"] = bat["runs"].where(
        bat["balls_faced_complete"], 0
    )

    bat["complete_bf_balls"] = bat[
        "reconstructed_balls_faced"
    ].where(
        bat["balls_faced_complete"], 0
    )

    sr_support = bat.groupby(
        bg,
        dropna=False
    ).agg(
        strike_rate_innings=("balls_faced_complete", "sum"),
        strike_rate_runs=("complete_bf_runs", "sum"),
        strike_rate_balls_faced=("complete_bf_balls", "sum")
    ).reset_index()

    batting = batting.merge(
        sr_support,
        on=bg,
        how="left",
        validate="one_to_one"
    )

    batting["strike_rate"] = (
        100 * batting["strike_rate_runs"]
        / batting["strike_rate_balls_faced"].replace(0, np.nan)
    )

    batting["strike_rate_coverage_pct"] = (
        100 * batting["strike_rate_innings"]
        / batting["innings_batted"].replace(0, np.nan)
    )

    batting["runs_per_innings"] = (
        batting["runs"]
        / batting["innings_batted"].replace(0, np.nan)
    )

    batting["batting_average"] = (
        batting["runs"]
        / batting["dismissals"].replace(0, np.nan)
    )

    batting["boundary_runs_pct"] = (
        100 * batting["boundary_runs"]
        / batting["runs"].replace(0, np.nan)
    )

    batting["runs_contribution_pct"] = (
        100 * batting["runs"]
        / batting["team_runs_in_batted_innings"].replace(0, np.nan)
    )

    # Metric-specific completeness.
    source_quality = bat.groupby(
        bg,
        dropna=False
    ).agg(
        missing_runs_source=("runs_source_missing", "sum"),
        missing_fours_source=("fours_source_missing", "sum"),
        missing_sixes_source=("sixes_source_missing", "sum"),
        missing_dismissals_source=("dismissals_source_missing", "sum")
    ).reset_index()

    batting = batting.merge(
        source_quality,
        on=bg,
        how="left",
        validate="one_to_one"
    )

    batting["runs_per_innings_complete"] = (
        batting["missing_runs_source"].eq(0)
    )

    batting["strike_rate_complete"] = (
        batting["incomplete_balls_faced_innings"].eq(0)
    )

    batting["strike_rate_available"] = (
        batting["strike_rate_balls_faced"].gt(0)
    )

    batting["batting_average_complete"] = (
        batting["missing_runs_source"].eq(0)
        & batting["missing_dismissals_source"].eq(0)
    )

    batting["boundary_runs_pct_complete"] = (
        batting["missing_runs_source"].eq(0)
        & batting["missing_fours_source"].eq(0)
        & batting["missing_sixes_source"].eq(0)
    )

    batting["runs_contribution_pct_complete"] = (
        batting["runs_per_innings_complete"]
    )

    batting["metric_data_complete"] = batting[
        [
            "runs_per_innings_complete",
            "strike_rate_complete",
            "batting_average_complete",
            "boundary_runs_pct_complete"
        ]
    ].all(axis=1)

    batting["eligibility_threshold_matches"] = np.where(
        batting["population"].eq("IPL"), 8, 12
    )

    batting["eligible_by_matches"] = (
        batting["matches_played"]
        >= batting["eligibility_threshold_matches"]
    )

    # ---------------------------------------------------------
    # BOWLING DELIVERY RECONSTRUCTION
    # ---------------------------------------------------------

    balls["extras_clean"] = clean(balls["extras_type"])

    e1 = balls["extras1_runs"]
    e2 = balls["extras_2_runs"]

    missing_type_with_extras = (
        balls["extras_clean"].eq("")
        & (e1.ne(0) | e2.ne(0))
    )

    ambiguous_type = balls["extras_clean"].isin(
        ["other", "penalty"]
    )

    unknown_type = ~balls["extras_clean"].isin(
        KNOWN_EXTRA_TYPES
    )

    balls["bowling_unresolved"] = (
        missing_type_with_extras
        | ambiguous_type
        | unknown_type
        | e2.ne(0)
    )

    # Wides and Overstep/no-balls are not legal balls.
    # Byes and leg-byes are legal but are not charged to the bowler.

    balls["is_legal"] = ~balls["extras_clean"].isin(
        ["wide", "overstep"]
    )

    balls["valid_bowling_delivery"] = (
        ~balls["bowling_unresolved"]
    )

    balls["valid_legal_ball"] = (
        balls["is_legal"]
        & balls["valid_bowling_delivery"]
    )

    balls["bowler_conceded_runs"] = (
        balls["runs"]
        + np.where(
            balls["extras_clean"].isin(["wide", "overstep"]),
            e1,
            0
        )
    )

    balls.loc[
        ~balls["valid_bowling_delivery"],
        "bowler_conceded_runs"
    ] = np.nan

    balls["dot_ball"] = (
        balls["valid_legal_ball"]
        & balls["team_runs_delivery"].eq(0)
    )

    dismissals = clean(balls["dismissal_type"])

    balls["bowler_wicket"] = (
        balls["valid_bowling_delivery"]
        & balls["is_wicket"].fillna(False).astype(bool)
        & dismissals.isin(BOWLER_CREDITED_DISMISSALS)
    ).astype(int)

    # Select IPL/T20I matches in the evaluation window.

    pop_matches = match_meta[
        [
            "match_id",
            "comp_id_y",
            "comp_name_meta",
            "match_format_meta",
            "season_meta"
        ]
    ].rename(
        columns={
            "comp_id_y": "comp_id",
            "comp_name_meta": "comp_name",
            "match_format_meta": "match_format",
            "season_meta": "season"
        }
    )

    pop_matches = classify_population(
        pop_matches.assign(
            match_date=pd.Timestamp("2022-01-01")
        )
    )

    match_dates = pd.concat(
        [
            bat[["match_id", "match_date"]],
            bowl_ref[["match_id", "match_date"]]
        ]
    ).drop_duplicates("match_id")

    pop_matches = pop_matches.drop(
        columns=["match_date"],
        errors="ignore"
    ).merge(
        match_dates,
        on="match_id",
        how="left",
        validate="one_to_one"
    )

    pop_matches = pop_matches[
        pd.to_datetime(
            pop_matches["match_date"],
            errors="coerce"
        ).between("2022-01-01", "2026-12-31")
    ]

    pop_ids = set(pop_matches["match_id"])

    balls = balls[
        balls["match_id"].isin(pop_ids)
    ].copy()

    balls = balls[
        balls["bowler_id"].notna()
    ].copy()

    balls["bowler_id"] = numeric(balls["bowler_id"])

    # Aggregate reconstructed bowling stats by player-match-innings.

    delivery_match = balls.groupby(
        ["match_id", "innings", "bowler_id"],
        as_index=False
    ).agg(
        valid_bowling_deliveries=("valid_bowling_delivery", "sum"),
        valid_legal_balls=("valid_legal_ball", "sum"),
        dot_balls=("dot_ball", "sum"),
        runs_conceded=(
            "bowler_conceded_runs",
            lambda x: x.sum(min_count=1)
        ),
        bowler_credited_wickets=("bowler_wicket", "sum"),
        ambiguous_deliveries=("bowling_unresolved", "sum"),
        all_deliveries=("bowler_id", "size")
    )

    bowl_ref["match_id"] = numeric(bowl_ref["match_id"])
    bowl_ref["bowler_id"] = numeric(bowl_ref["bowler_id"])

    delivery_match = delivery_match.merge(
        bowl_ref[
            [
                "match_id",
                "innings",
                "bowler_id",
                "bowler_name",
                "population",
                "evaluation_period",
                "match_date"
            ]
        ].drop_duplicates(bowler_key),
        on=["match_id", "innings", "bowler_id"],
        how="left",
        validate="one_to_one"
    )

    if delivery_match["population"].isna().any():
        raise ValueError(
            "Some reconstructed bowling delivery groups do not match player_match_bowling."
        )

    wg = [
        "population",
        "evaluation_period",
        "bowler_id",
        "bowler_name"
    ]

    bowling = delivery_match.groupby(
        wg,
        dropna=False
    ).agg(
        matches_played=("match_id", "nunique"),
        bowling_innings=("innings", "count"),
        runs_conceded=("runs_conceded", "sum"),
        valid_bowling_deliveries=("valid_bowling_deliveries", "sum"),
        legal_balls=("valid_legal_balls", "sum"),
        wickets=("bowler_credited_wickets", "sum"),
        dot_balls=("dot_balls", "sum"),
        ambiguous_deliveries=("ambiguous_deliveries", "sum"),
        first_match=("match_date", "min"),
        last_match=("match_date", "max")
    ).reset_index()

    bowling["wickets_per_innings"] = (
        bowling["wickets"]
        / bowling["bowling_innings"].replace(0, np.nan)
    )

    bowling["dot_ball_pct"] = (
        100 * bowling["dot_balls"]
        / bowling["legal_balls"].replace(0, np.nan)
    )

    bowling["economy"] = (
        6 * bowling["runs_conceded"]
        / bowling["legal_balls"].replace(0, np.nan)
    )

    bowling["bowling_average"] = (
        bowling["runs_conceded"]
        / bowling["wickets"].replace(0, np.nan)
    )

    bowling["bowling_strike_rate"] = (
        bowling["legal_balls"]
        / bowling["wickets"].replace(0, np.nan)
    )

    bowling["valid_delivery_coverage_pct"] = (
        100 * bowling["valid_bowling_deliveries"]
        / (
            bowling["valid_bowling_deliveries"]
            + bowling["ambiguous_deliveries"]
        ).replace(0, np.nan)
    )

    # Availability and completeness are separate concepts.

    bowling["wickets_per_innings_available"] = (
        bowling["bowling_innings"].gt(0)
    )

    bowling["dot_ball_pct_available"] = (
        bowling["legal_balls"].gt(0)
    )

    bowling["economy_available"] = (
        bowling["legal_balls"].gt(0)
    )

    bowling["bowling_average_available"] = (
        bowling["wickets"].gt(0)
    )

    bowling["bowling_strike_rate_available"] = (
        bowling["wickets"].gt(0)
    )

    no_ambiguous_bowling = (
        bowling["ambiguous_deliveries"].eq(0)
    )

    bowling["wickets_per_innings_complete"] = (
        no_ambiguous_bowling
        & bowling["wickets_per_innings_available"]
    )

    bowling["dot_ball_pct_complete"] = (
        no_ambiguous_bowling
        & bowling["dot_ball_pct_available"]
    )

    bowling["economy_complete"] = (
        no_ambiguous_bowling
        & bowling["economy_available"]
    )

    bowling["bowling_average_complete"] = (
        no_ambiguous_bowling
        & bowling["bowling_average_available"]
    )

    bowling["bowling_strike_rate_complete"] = (
        no_ambiguous_bowling
        & bowling["bowling_strike_rate_available"]
    )

    bowling["metric_data_complete"] = bowling[
        [
            "wickets_per_innings_complete",
            "dot_ball_pct_complete",
            "economy_complete",
            "bowling_average_complete"
        ]
    ].all(axis=1)

    bowling["eligibility_threshold_matches"] = np.where(
        bowling["population"].eq("IPL"), 8, 12
    )

    bowling["eligible_by_matches"] = (
        bowling["matches_played"]
        >= bowling["eligibility_threshold_matches"]
    )

    # ---------------------------------------------------------
    # BOWLING RECONCILIATION
    # ---------------------------------------------------------

    ref = bowl_ref.copy()

    ref["match_id"] = numeric(ref["match_id"])
    ref["bowler_id"] = numeric(ref["bowler_id"])
    ref["innings"] = numeric(ref["innings"])
    ref["legal_balls"] = numeric(ref["legal_balls"])
    ref["runs_conceded"] = numeric(ref["runs_conceded"])
    ref["wickets"] = numeric(ref["wickets"])

    recon = delivery_match.merge(
        ref[
            [
                "match_id",
                "innings",
                "bowler_id",
                "legal_balls",
                "runs_conceded",
                "wickets",
                "unresolved_deliveries",
                "population",
                "evaluation_period"
            ]
        ],
        on=["match_id", "innings", "bowler_id"],
        how="left",
        validate="one_to_one",
        suffixes=("_reconstructed", "_reference")
    )

    recon["legal_ball_difference"] = (
        recon["valid_legal_balls"] - recon["legal_balls"]
    )

    recon["runs_conceded_difference"] = (
        recon["runs_conceded_reconstructed"]
        - recon["runs_conceded_reference"]
    )

    recon["wicket_difference"] = (
        recon["bowler_credited_wickets"] - recon["wickets"]
    )

    recon_summary = recon.groupby(
        [
            "population_reconstructed",
            "evaluation_period_reconstructed"
        ],
        as_index=False
    ).agg(
        player_match_innings=("match_id", "size"),
        legal_ball_mismatch_rows=(
            "legal_ball_difference",
            lambda x: int(x.ne(0).sum())
        ),
        absolute_legal_ball_difference=(
            "legal_ball_difference",
            lambda x: float(x.abs().sum())
        ),
        runs_conceded_mismatch_rows=(
            "runs_conceded_difference",
            lambda x: int((x.abs() > 1e-9).sum())
        ),
        absolute_runs_conceded_difference=(
            "runs_conceded_difference",
            lambda x: float(x.abs().sum())
        ),
        wicket_mismatch_rows=(
            "wicket_difference",
            lambda x: int(x.ne(0).sum())
        ),
        absolute_wicket_difference=(
            "wicket_difference",
            lambda x: float(x.abs().sum())
        ),
        ambiguous_deliveries=("ambiguous_deliveries", "sum"),
        reconstructed_dot_balls=("dot_balls", "sum"),
        reconstructed_valid_legal_balls=("valid_legal_balls", "sum")
    ).rename(
        columns={
            "population_reconstructed": "population",
            "evaluation_period_reconstructed": "evaluation_period"
        }
    )

    # ---------------------------------------------------------
    # DIAGNOSTICS
    # ---------------------------------------------------------

    batting_vars = [
        "runs_per_innings",
        "strike_rate",
        "batting_average",
        "boundary_runs_pct",
        "runs_contribution_pct"
    ]

    bowling_vars = [
        "wickets_per_innings",
        "dot_ball_pct",
        "economy",
        "bowling_average",
        "bowling_strike_rate"
    ]

    diagnostics = []

    for role, frame, metrics in [
        ("Batting", batting, batting_vars),
        ("Bowling", bowling, bowling_vars)
    ]:

        for (pop, period), group in frame.groupby(
            ["population", "evaluation_period"]
        ):

            eligible = group[group["eligible_by_matches"]]

            for metric in metrics:

                vals = pd.to_numeric(
                    group[metric],
                    errors="coerce"
                )

                elig_vals = pd.to_numeric(
                    eligible[metric],
                    errors="coerce"
                )

                diagnostics.append({
                    "role": role,
                    "population": pop,
                    "evaluation_period": period,
                    "metric": metric,
                    "player_units": len(group),
                    "eligible_player_units": len(eligible),
                    "non_missing": int(vals.notna().sum()),
                    "missing": int(vals.isna().sum()),
                    "missing_pct": round(
                        100 * vals.isna().mean(), 2
                    ),
                    "eligible_non_missing": int(
                        elig_vals.notna().sum()
                    ),
                    "min": vals.min(),
                    "p05": vals.quantile(.05),
                    "median": vals.median(),
                    "mean": vals.mean(),
                    "p95": vals.quantile(.95),
                    "max": vals.max(),
                    "eligible_median": elig_vals.median()
                })

    diagnostics = pd.DataFrame(diagnostics)

    correlations = []

    for role, frame, metrics in [
        ("Batting", batting, batting_vars),
        ("Bowling", bowling, bowling_vars)
    ]:

        for (pop, period), group in frame.groupby(
            ["population", "evaluation_period"]
        ):

            eligible = group[group["eligible_by_matches"]]
            corr = eligible[metrics].corr(method="spearman")

            for i, m1 in enumerate(metrics):

                for m2 in metrics[i + 1:]:

                    correlations.append({
                        "role": role,
                        "population": pop,
                        "evaluation_period": period,
                        "metric_1": m1,
                        "metric_2": m2,
                        "spearman_correlation": corr.loc[m1, m2]
                    })

    correlations = pd.DataFrame(correlations)

    validation = pd.DataFrame([
        (
            "Batting player-population units",
            len(batting)
        ),
        (
            "Bowling player-population units",
            len(bowling)
        ),
        (
            "Eligible batting units",
            int(batting["eligible_by_matches"].sum())
        ),
        (
            "Eligible bowling units",
            int(bowling["eligible_by_matches"].sum())
        ),
        (
            "Duplicate batter match-innings keys",
            0
        ),
        (
            "Duplicate bowler match-innings keys",
            0
        ),
        (
            "Ambiguous bowling deliveries excluded from all reconstructed bowling metrics",
            int(balls["bowling_unresolved"].sum())
        ),
        (
            "Ambiguous batting deliveries not guessed for balls-faced reconstruction",
            int(balls["batting_delivery_ambiguous"].sum())
        ),
        (
            "Batter match-innings with balls-faced mismatch",
            int(bat["balls_faced_difference"].ne(0).sum())
        ),
        (
            "Batter match-innings with incomplete reconstructed balls faced",
            int((~bat["balls_faced_complete"]).sum())
        ),
        (
            "Batting strike rate uses complete innings only",
            "Runs and balls faced both restricted to reconciled innings"
        ),
        (
            "Batting strike-rate eligible units with at least one complete innings",
            int(
                (
                    batting["eligible_by_matches"]
                    & batting["strike_rate_available"]
                ).sum()
            )
        ),
        (
            "Bowling metric availability is separate from full-source completeness",
            "See *_available, *_complete and valid_delivery_coverage_pct fields"
        ),
        (
            "Bowling dot-ball denominator",
            "Reconstructed valid legal deliveries only"
        ),
        (
            "Bowling runs conceded",
            "Batter runs plus wides/Overstep extras on valid deliveries"
        ),
        (
            "Bowler-credited wickets",
            "Caught, Bowled, LBW, Stump, Hit Wicket on valid deliveries"
        ),
        (
            "First evaluated match date",
            str(pd.to_datetime(bat["match_date"]).min().date())
        ),
        (
            "Last evaluated match date",
            str(pd.to_datetime(bat["match_date"]).max().date())
        )
    ], columns=["check", "value"])

    # ---------------------------------------------------------
    # SAVE OUTPUTS
    # ---------------------------------------------------------

    batting.to_csv(
        OUT / "player_batting_metrics.csv",
        index=False
    )

    bowling.to_csv(
        OUT / "player_bowling_metrics.csv",
        index=False
    )

    diagnostics.to_csv(
        OUT / "player_metric_diagnostics.csv",
        index=False
    )

    correlations.to_csv(
        OUT / "player_metric_correlations.csv",
        index=False
    )

    recon_summary.to_csv(
        OUT / "bowling_dot_ball_validation.csv",
        index=False
    )

    validation.to_csv(
        OUT / "player_metric_validation.csv",
        index=False
    )

    recon.to_csv(
        OUT / "bowling_match_innings_reconciliation.csv",
        index=False
    )

    bat[
        [
            "match_id",
            "innings",
            "striker_id",
            "striker_name",
            "population",
            "evaluation_period",
            "balls_faced",
            "reconstructed_balls_faced",
            "balls_faced_difference",
            "ambiguous_batting_deliveries",
            "unresolved_deliveries",
            "balls_faced_complete"
        ]
    ].to_csv(
        OUT / "batting_match_innings_reconciliation.csv",
        index=False
    )

    # ---------------------------------------------------------
    # TERMINAL SUMMARY
    # ---------------------------------------------------------

    print("\nCORRECTED PLAYER METRIC BUILD COMPLETE")
    print(f"Project root: {ROOT}")
    print(f"Outputs: {OUT}")

    print("\nBATTER POPULATION SUMMARY")

    print(
        batting.groupby(
            ["population", "evaluation_period"]
        ).agg(
            player_units=("striker_id", "nunique"),
            eligible=("eligible_by_matches", "sum")
        ).to_string()
    )

    print("\nBOWLER POPULATION SUMMARY")

    print(
        bowling.groupby(
            ["population", "evaluation_period"]
        ).agg(
            player_units=("bowler_id", "nunique"),
            eligible=("eligible_by_matches", "sum")
        ).to_string()
    )

    print("\nBATTING BALLS-FACED RECONCILIATION")

    print(
        bat.groupby(
            ["population", "evaluation_period"]
        ).agg(
            batter_innings=("match_id", "size"),
            balls_faced_mismatch_innings=(
                "balls_faced_difference",
                lambda x: int(x.ne(0).sum())
            ),
            incomplete_balls_faced_innings=(
                "balls_faced_complete",
                lambda x: int((~x).sum())
            ),
            ambiguous_deliveries=(
                "ambiguous_batting_deliveries",
                "sum"
            )
        ).to_string()
    )

    print("\nBOWLING RECONCILIATION")
    print(recon_summary.to_string(index=False))

    print("\nVALIDATION")
    print(validation.to_string(index=False))

    print(
        "\nSaved metric-specific completeness flags, "
        "batting/bowling reconciliations and diagnostics."
    )


if __name__ == "__main__":
    main()