from pathlib import Path
import numpy as np
import pandas as pd

# SupeRR Selector Season 2 — Round 1
# Pressure Performance Diagnostics
#
# Diagnostic outputs only. This script does NOT assign final weights,
# calculate a Pressure Performance Index, or combine IPL and T20I scores.

def find_root():
    here = Path(__file__).resolve()
    for candidate in [Path.cwd(), *here.parents]:
        if (
            (candidate / "Data Pack").exists()
            and (candidate / "outputs" / "processed").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "Project root not found. Put this script in SupeRR/src and run it from the project."
    )


ROOT = find_root()
DATA = ROOT / "Data Pack"
PROC = ROOT / "outputs" / "processed"
OUT = ROOT / "outputs" / "analysis" / "pressure_index_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

PATHS = {
    "states": PROC / "pressure_states.csv",
    "batting": PROC / "player_match_pressure_batting.csv",
    "bowling": PROC / "player_match_pressure_bowling.csv",
}

REQUIRED = {
    "states": [
        "match_id", "match_date", "competition_group", "season", "innings",
        "overid", "over_ball_num", "striker_id", "bowler_id",
        "pressure_eligible", "chase_prediction_supported",
        "batting_pressure", "bowling_pressure",
    ],
    "batting": [
        "match_id", "match_date", "competition_group", "season", "innings",
        "phase", "striker_id", "striker_name", "pressure_level",
        "pressure_deliveries", "balls_faced", "batter_runs", "fours",
        "sixes", "dismissals", "ambiguous_extras_deliveries",
        "batting_stats_complete",
    ],
    "bowling": [
        "match_id", "match_date", "competition_group", "season", "innings",
        "phase", "bowler_id", "bowler_name", "pressure_level",
        "pressure_deliveries", "valid_bowling_deliveries", "legal_balls",
        "runs_conceded", "wickets", "ambiguous_deliveries",
        "bowling_stats_complete",
    ],
}


def require(df, cols, label):
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f"{label}: missing required columns: {missing}")


def clean_bool(s):
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(
        ["1", "true", "t", "yes", "y"]
    )


def numeric(s):
    return pd.to_numeric(s, errors="coerce")


def add_period(df):
    d = df.copy()
    d["competition_group"] = d["competition_group"].astype(str).str.strip()
    d["season"] = d["season"].astype(str).str.strip()
    d["evaluation_period"] = np.where(
        d["competition_group"].eq("IPL"),
        d["season"],
        "2022-2026 combined",
    )
    d["match_id"] = numeric(d["match_id"]).astype("Int64")
    d["innings"] = numeric(d["innings"])
    d["match_date"] = pd.to_datetime(d["match_date"], errors="coerce")
    return d


def aggregate_batting(bat):
    d = bat.copy()
    for c in [
        "pressure_deliveries", "balls_faced", "batter_runs", "fours",
        "sixes", "dismissals", "ambiguous_extras_deliveries",
    ]:
        d[c] = numeric(d[c]).fillna(0)
    d["batting_stats_complete"] = clean_bool(d["batting_stats_complete"])
    d["pressure_level"] = d["pressure_level"].astype(str).str.strip().str.title()
    d["is_high"] = d["pressure_level"].eq("High")
    d["is_baseline"] = d["pressure_level"].isin(["Low", "Moderate"])

    keys = [
        "competition_group", "evaluation_period", "striker_id", "striker_name",
        "innings", "phase",
    ]

    # High-pressure and non-high-pressure totals within the same innings/phase.
    def sum_where(frame, mask, col):
        return frame[col].where(mask, 0)

    d["high_balls"] = sum_where(d, d["is_high"], "balls_faced")
    d["high_runs"] = sum_where(d, d["is_high"], "batter_runs")
    d["high_fours"] = sum_where(d, d["is_high"], "fours")
    d["high_sixes"] = sum_where(d, d["is_high"], "sixes")
    d["high_dismissals"] = sum_where(d, d["is_high"], "dismissals")
    d["high_matches"] = d["match_id"].where(d["is_high"])
    d["high_ambiguous"] = sum_where(
        d, d["is_high"], "ambiguous_extras_deliveries"
    )

    # Exclude incomplete aggregate rows from rate calculations, but preserve
    # their exposure and ambiguity counts in the diagnostic outputs.
    complete_high = d["is_high"] & d["batting_stats_complete"]
    complete_base = d["is_baseline"] & d["batting_stats_complete"]
    d["high_complete_balls"] = sum_where(d, complete_high, "balls_faced")
    d["high_complete_runs"] = sum_where(d, complete_high, "batter_runs")
    d["base_complete_balls"] = sum_where(d, complete_base, "balls_faced")
    d["base_complete_runs"] = sum_where(d, complete_base, "batter_runs")
    d["base_complete_dismissals"] = sum_where(d, complete_base, "dismissals")
    d["base_complete_fours"] = sum_where(d, complete_base, "fours")
    d["base_complete_sixes"] = sum_where(d, complete_base, "sixes")
    d["high_complete_dismissals"] = sum_where(
        d, complete_high, "dismissals"
    )

    strata = d.groupby(keys, dropna=False).agg(
        high_pressure_matches=("high_matches", "nunique"),
        high_pressure_deliveries=("pressure_deliveries", lambda x: x[d.loc[x.index, "is_high"]].sum()),
        high_balls_faced=("high_balls", "sum"),
        high_runs=("high_runs", "sum"),
        high_fours=("high_fours", "sum"),
        high_sixes=("high_sixes", "sum"),
        high_dismissals=("high_dismissals", "sum"),
        high_ambiguous_deliveries=("high_ambiguous", "sum"),
        high_complete_balls=("high_complete_balls", "sum"),
        high_complete_runs=("high_complete_runs", "sum"),
        high_complete_dismissals=("high_complete_dismissals", "sum"),
        baseline_balls=("base_complete_balls", "sum"),
        baseline_runs=("base_complete_runs", "sum"),
        baseline_dismissals=("base_complete_dismissals", "sum"),
        baseline_fours=("base_complete_fours", "sum"),
        baseline_sixes=("base_complete_sixes", "sum"),
        total_pressure_rows=("match_id", "size"),
    ).reset_index()

    strata["high_pressure_strike_rate"] = (
        100 * strata["high_complete_runs"]
        / strata["high_complete_balls"].replace(0, np.nan)
    )
    strata["baseline_strike_rate"] = (
        100 * strata["baseline_runs"]
        / strata["baseline_balls"].replace(0, np.nan)
    )
    strata["strike_rate_uplift"] = (
        strata["high_pressure_strike_rate"] - strata["baseline_strike_rate"]
    )
    strata["high_pressure_average"] = (
        strata["high_complete_runs"]
        / strata["high_complete_dismissals"].replace(0, np.nan)
    )
    strata["baseline_average"] = (
        strata["baseline_runs"]
        / strata["baseline_dismissals"].replace(0, np.nan)
    )
    strata["average_uplift"] = (
        strata["high_pressure_average"] - strata["baseline_average"]
    )
    strata["high_boundary_run_pct"] = (
        100 * (4 * strata["high_fours"] + 6 * strata["high_sixes"])
        / strata["high_runs"].replace(0, np.nan)
    )

    player_keys = [
        "competition_group", "evaluation_period", "striker_id", "striker_name"
    ]
    player = strata.groupby(player_keys, dropna=False).agg(
        high_pressure_matches_strata_sum=("high_pressure_matches", "sum"),
        high_pressure_deliveries=("high_pressure_deliveries", "sum"),
        high_balls_faced=("high_balls_faced", "sum"),
        high_runs=("high_runs", "sum"),
        high_dismissals=("high_dismissals", "sum"),
        high_ambiguous_deliveries=("high_ambiguous_deliveries", "sum"),
        high_complete_balls=("high_complete_balls", "sum"),
        high_complete_runs=("high_complete_runs", "sum"),
        high_complete_dismissals=("high_complete_dismissals", "sum"),
        baseline_balls=("baseline_balls", "sum"),
        baseline_runs=("baseline_runs", "sum"),
        baseline_dismissals=("baseline_dismissals", "sum"),
    ).reset_index()

    player["high_pressure_strike_rate"] = (
        100 * player["high_complete_runs"]
        / player["high_complete_balls"].replace(0, np.nan)
    )
    player["high_pressure_average"] = (
        player["high_complete_runs"]
        / player["high_complete_dismissals"].replace(0, np.nan)
    )
    player["baseline_strike_rate"] = (
        100 * player["baseline_runs"]
        / player["baseline_balls"].replace(0, np.nan)
    )
    player["baseline_average"] = (
        player["baseline_runs"]
        / player["baseline_dismissals"].replace(0, np.nan)
    )

    # Uplift is calculated within matching innings/phase strata, then weighted
    # by high-pressure balls. Strata without both high and baseline evidence
    # do not contribute to uplift.
    uplift = strata.dropna(subset=["strike_rate_uplift"]).copy()
    uplift["weighted_sr_uplift"] = (
        uplift["strike_rate_uplift"] * uplift["high_complete_balls"]
    )
    uplift_player = uplift.groupby(player_keys, dropna=False).agg(
        sr_uplift_n_strata=("strike_rate_uplift", "count"),
        sr_uplift_weight=("high_complete_balls", "sum"),
        sr_uplift_weighted_sum=("weighted_sr_uplift", "sum"),
    ).reset_index()
    uplift_player["pressure_sr_uplift"] = (
        uplift_player["sr_uplift_weighted_sum"]
        / uplift_player["sr_uplift_weight"].replace(0, np.nan)
    )
    player = player.merge(
        uplift_player[player_keys + ["sr_uplift_n_strata", "pressure_sr_uplift"]],
        on=player_keys, how="left", validate="one_to_one",
    )

    # Match-level high-pressure strike rates for consistency diagnostics.
    m = d[d["is_high"] & d["batting_stats_complete"]].groupby(
        player_keys + ["match_id"], dropna=False
    ).agg(
        match_high_balls=("balls_faced", "sum"),
        match_high_runs=("batter_runs", "sum"),
    ).reset_index()
    m["match_high_strike_rate"] = (
        100 * m["match_high_runs"] / m["match_high_balls"].replace(0, np.nan)
    )
    consistency = m.groupby(player_keys, dropna=False).agg(
        high_pressure_match_count_for_consistency=("match_id", "nunique"),
        match_sr_mean=("match_high_strike_rate", "mean"),
        match_sr_sd=("match_high_strike_rate", "std"),
        match_sr_median=("match_high_strike_rate", "median"),
        match_sr_mad=("match_high_strike_rate", lambda x: (x - x.median()).abs().median()),
    ).reset_index()
    player = player.merge(consistency, on=player_keys, how="left", validate="one_to_one")
    distinct_high_matches = d[d["is_high"]].groupby(
        player_keys, dropna=False
    )["match_id"].nunique().rename("high_pressure_matches").reset_index()
    player = player.merge(
        distinct_high_matches, on=player_keys, how="left", validate="one_to_one"
    )
    player = player.drop(columns=["high_pressure_matches_strata_sum"])

    player["high_pressure_match_support_band"] = pd.cut(
        player["high_pressure_matches"],
        bins=[-0.1, 2, 5, np.inf],
        labels=["Low (1-2)", "Moderate (3-5)", "Higher (6+)"],
    )
    player["high_pressure_sr_coverage_pct"] = (
        100 * player["high_complete_balls"]
        / player["high_balls_faced"].replace(0, np.nan)
    )
    return player, strata, m


def aggregate_bowling(bowl):
    d = bowl.copy()
    for c in [
        "pressure_deliveries", "valid_bowling_deliveries", "legal_balls",
        "runs_conceded", "wickets", "ambiguous_deliveries",
    ]:
        d[c] = numeric(d[c]).fillna(0)
    d["bowling_stats_complete"] = clean_bool(d["bowling_stats_complete"])
    d["pressure_level"] = d["pressure_level"].astype(str).str.strip().str.title()
    d["is_high"] = d["pressure_level"].eq("High")
    d["is_baseline"] = d["pressure_level"].isin(["Low", "Moderate"])

    keys = [
        "competition_group", "evaluation_period", "bowler_id", "bowler_name",
        "innings", "phase",
    ]
    d["high_matches"] = d["match_id"].where(d["is_high"])
    for col in ["pressure_deliveries", "valid_bowling_deliveries",
                "legal_balls", "runs_conceded", "wickets", "ambiguous_deliveries"]:
        d[f"high_{col}"] = d[col].where(d["is_high"], 0)
        d[f"base_{col}"] = d[col].where(d["is_baseline"], 0)

    strata = d.groupby(keys, dropna=False).agg(
        high_pressure_matches=("high_matches", "nunique"),
        high_pressure_deliveries=("high_pressure_deliveries", "sum"),
        high_valid_deliveries=("high_valid_bowling_deliveries", "sum"),
        high_legal_balls=("high_legal_balls", "sum"),
        high_runs_conceded=("high_runs_conceded", "sum"),
        high_wickets=("high_wickets", "sum"),
        high_ambiguous_deliveries=("high_ambiguous_deliveries", "sum"),
        baseline_valid_deliveries=("base_valid_bowling_deliveries", "sum"),
        baseline_legal_balls=("base_legal_balls", "sum"),
        baseline_runs_conceded=("base_runs_conceded", "sum"),
        baseline_wickets=("base_wickets", "sum"),
    ).reset_index()

    strata["high_economy"] = (
        6 * strata["high_runs_conceded"]
        / strata["high_legal_balls"].replace(0, np.nan)
    )
    strata["baseline_economy"] = (
        6 * strata["baseline_runs_conceded"]
        / strata["baseline_legal_balls"].replace(0, np.nan)
    )
    # Positive value means lower economy under high pressure.
    strata["economy_uplift"] = (
        strata["baseline_economy"] - strata["high_economy"]
    )
    strata["high_wicket_rate_per_6"] = (
        6 * strata["high_wickets"]
        / strata["high_legal_balls"].replace(0, np.nan)
    )
    strata["baseline_wicket_rate_per_6"] = (
        6 * strata["baseline_wickets"]
        / strata["baseline_legal_balls"].replace(0, np.nan)
    )
    strata["wicket_rate_uplift"] = (
        strata["high_wicket_rate_per_6"]
        - strata["baseline_wicket_rate_per_6"]
    )
    strata["high_bowling_average"] = (
        strata["high_runs_conceded"]
        / strata["high_wickets"].replace(0, np.nan)
    )

    player_keys = [
        "competition_group", "evaluation_period", "bowler_id", "bowler_name"
    ]
    player = strata.groupby(player_keys, dropna=False).agg(
        high_pressure_matches_strata_sum=("high_pressure_matches", "sum"),
        high_pressure_deliveries=("high_pressure_deliveries", "sum"),
        high_valid_deliveries=("high_valid_deliveries", "sum"),
        high_legal_balls=("high_legal_balls", "sum"),
        high_runs_conceded=("high_runs_conceded", "sum"),
        high_wickets=("high_wickets", "sum"),
        high_ambiguous_deliveries=("high_ambiguous_deliveries", "sum"),
        baseline_legal_balls=("baseline_legal_balls", "sum"),
        baseline_runs_conceded=("baseline_runs_conceded", "sum"),
        baseline_wickets=("baseline_wickets", "sum"),
    ).reset_index()

    player["high_pressure_economy"] = (
        6 * player["high_runs_conceded"]
        / player["high_legal_balls"].replace(0, np.nan)
    )
    player["high_pressure_bowling_average"] = (
        player["high_runs_conceded"]
        / player["high_wickets"].replace(0, np.nan)
    )
    player["high_pressure_wickets_per_6_legal_balls"] = (
        6 * player["high_wickets"]
        / player["high_legal_balls"].replace(0, np.nan)
    )
    player["baseline_economy"] = (
        6 * player["baseline_runs_conceded"]
        / player["baseline_legal_balls"].replace(0, np.nan)
    )
    player["baseline_wickets_per_6_legal_balls"] = (
        6 * player["baseline_wickets"]
        / player["baseline_legal_balls"].replace(0, np.nan)
    )

    # Matched-strata uplift: compare high pressure to Low+Moderate within the
    # same innings and phase; weight by high-pressure legal balls.
    valid_e = strata.dropna(subset=["economy_uplift"]).copy()
    valid_e["weighted_economy_uplift"] = (
        valid_e["economy_uplift"] * valid_e["high_legal_balls"]
    )
    eu = valid_e.groupby(player_keys, dropna=False).agg(
        economy_uplift_weight=("high_legal_balls", "sum"),
        economy_uplift_sum=("weighted_economy_uplift", "sum"),
        economy_uplift_strata=("economy_uplift", "count"),
    ).reset_index()
    eu["pressure_economy_uplift"] = (
        eu["economy_uplift_sum"]
        / eu["economy_uplift_weight"].replace(0, np.nan)
    )

    valid_w = strata.dropna(subset=["wicket_rate_uplift"]).copy()
    valid_w["weighted_wicket_uplift"] = (
        valid_w["wicket_rate_uplift"] * valid_w["high_legal_balls"]
    )
    wu = valid_w.groupby(player_keys, dropna=False).agg(
        wicket_uplift_sum=("weighted_wicket_uplift", "sum"),
        wicket_uplift_strata=("wicket_rate_uplift", "count"),
    ).reset_index()
    wu["pressure_wicket_rate_uplift"] = (
        wu["wicket_uplift_sum"]
        / valid_w.groupby(player_keys, dropna=False)["high_legal_balls"].sum().values
    )
    player = player.merge(
        eu[player_keys + ["pressure_economy_uplift", "economy_uplift_strata"]],
        on=player_keys, how="left", validate="one_to_one",
    )
    player = player.merge(
        wu[player_keys + ["pressure_wicket_rate_uplift", "wicket_uplift_strata"]],
        on=player_keys, how="left", validate="one_to_one",
    )

    # Match-level high-pressure economy for consistency diagnostics.
    m = d[d["is_high"]].groupby(
        player_keys + ["match_id"], dropna=False
    ).agg(
        match_high_legal_balls=("legal_balls", "sum"),
        match_high_runs_conceded=("runs_conceded", "sum"),
        match_high_wickets=("wickets", "sum"),
    ).reset_index()
    m["match_high_economy"] = (
        6 * m["match_high_runs_conceded"]
        / m["match_high_legal_balls"].replace(0, np.nan)
    )
    consistency = m.groupby(player_keys, dropna=False).agg(
        high_pressure_match_count_for_consistency=("match_id", "nunique"),
        match_economy_mean=("match_high_economy", "mean"),
        match_economy_sd=("match_high_economy", "std"),
        match_economy_median=("match_high_economy", "median"),
        match_economy_mad=("match_high_economy", lambda x: (x - x.median()).abs().median()),
    ).reset_index()
    player = player.merge(consistency, on=player_keys, how="left", validate="one_to_one")
    distinct_high_matches = d[d["is_high"]].groupby(
        player_keys, dropna=False
    )["match_id"].nunique().rename("high_pressure_matches").reset_index()
    player = player.merge(
        distinct_high_matches, on=player_keys, how="left", validate="one_to_one"
    )
    player = player.drop(columns=["high_pressure_matches_strata_sum"])

    player["high_pressure_match_support_band"] = pd.cut(
        player["high_pressure_matches"],
        bins=[-0.1, 2, 5, np.inf],
        labels=["Low (1-2)", "Moderate (3-5)", "Higher (6+)"],
    )
    player["high_pressure_delivery_coverage_pct"] = (
        100 * player["high_valid_deliveries"]
        / player["high_pressure_deliveries"].replace(0, np.nan)
    )
    return player, strata, m


def main():
    for label, path in PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} input: {path}")

    states = pd.read_csv(PATHS["states"], low_memory=False)
    bat = pd.read_csv(PATHS["batting"], low_memory=False)
    bowl = pd.read_csv(PATHS["bowling"], low_memory=False)

    for label, df in [("states", states), ("batting", bat), ("bowling", bowl)]:
        require(df, REQUIRED[label], label)

    states = add_period(states)
    bat = add_period(bat)
    bowl = add_period(bowl)

    # T20I is deliberately combined across the 2022-2026 evaluation window;
    # pressure coverage itself begins in 2024. No pre-2024 pressure zeros added.
    bat = bat[bat["competition_group"].isin(["IPL", "T20I"])].copy()
    bowl = bowl[bowl["competition_group"].isin(["IPL", "T20I"])].copy()

    batting_player, batting_strata, batting_matches = aggregate_batting(bat)
    bowling_player, bowling_strata, bowling_matches = aggregate_bowling(bowl)

    # Exposure summary by population and period.
    exposure_b = batting_player.groupby(
        ["competition_group", "evaluation_period"], dropna=False
    ).agg(
        players_with_pressure_data=("striker_id", "nunique"),
        median_high_pressure_matches=("high_pressure_matches", "median"),
        players_1_2_high_matches=("high_pressure_match_support_band", lambda x: int((x == "Low (1-2)").sum())),
        players_3_5_high_matches=("high_pressure_match_support_band", lambda x: int((x == "Moderate (3-5)").sum())),
        players_6plus_high_matches=("high_pressure_match_support_band", lambda x: int((x == "Higher (6+)").sum())),
        median_high_pressure_balls=("high_balls_faced", "median"),
        median_high_pressure_sr_coverage=("high_pressure_sr_coverage_pct", "median"),
    ).reset_index().rename(columns={"competition_group": "population"})

    exposure_w = bowling_player.groupby(
        ["competition_group", "evaluation_period"], dropna=False
    ).agg(
        players_with_pressure_data=("bowler_id", "nunique"),
        median_high_pressure_matches=("high_pressure_matches", "median"),
        players_1_2_high_matches=("high_pressure_match_support_band", lambda x: int((x == "Low (1-2)").sum())),
        players_3_5_high_matches=("high_pressure_match_support_band", lambda x: int((x == "Moderate (3-5)").sum())),
        players_6plus_high_matches=("high_pressure_match_support_band", lambda x: int((x == "Higher (6+)").sum())),
        median_high_pressure_legal_balls=("high_legal_balls", "median"),
        median_high_pressure_delivery_coverage=("high_pressure_delivery_coverage_pct", "median"),
    ).reset_index().rename(columns={"competition_group": "population"})

    # Structural validation.
    bat_dup = int(bat.duplicated(
        ["match_id", "innings", "phase", "striker_id", "pressure_level"]
    ).sum())
    bowl_dup = int(bowl.duplicated(
        ["match_id", "innings", "phase", "bowler_id", "pressure_level"]
    ).sum())

    state_eligible = clean_bool(states["pressure_eligible"])
    chase_supported = clean_bool(states["chase_prediction_supported"])
    state_summary = pd.DataFrame([
        ("Pressure-state rows", len(states)),
        ("Pressure-eligible state rows", int(state_eligible.sum())),
        ("Chase prediction-supported state rows", int(chase_supported.sum())),
        ("Pressure-state matches", int(states["match_id"].nunique())),
        ("Pressure-state first date", str(states["match_date"].min())),
        ("Pressure-state last date", str(states["match_date"].max())),
        ("Duplicate batting pressure aggregate keys", bat_dup),
        ("Duplicate bowling pressure aggregate keys", bowl_dup),
        ("Batting aggregate incomplete rows", int((~clean_bool(bat["batting_stats_complete"])).sum())),
        ("Bowling aggregate incomplete rows", int((~clean_bool(bowl["bowling_stats_complete"])).sum())),
        ("Batting High-pressure player units", len(batting_player)),
        ("Bowling High-pressure player units", len(bowling_player)),
        ("Final Pressure Performance Index created", "No — diagnostics only"),
        ("Final component weights assigned", "No"),
        ("Pressure-specific dot-ball metric", "Not calculated in this script"),
    ], columns=["check", "value"])

    batting_player.to_csv(OUT / "batting_pressure_player_diagnostics.csv", index=False)
    bowling_player.to_csv(OUT / "bowling_pressure_player_diagnostics.csv", index=False)
    batting_strata.to_csv(OUT / "batting_pressure_innings_phase_diagnostics.csv", index=False)
    bowling_strata.to_csv(OUT / "bowling_pressure_innings_phase_diagnostics.csv", index=False)
    batting_matches.to_csv(OUT / "batting_high_pressure_match_diagnostics.csv", index=False)
    bowling_matches.to_csv(OUT / "bowling_high_pressure_match_diagnostics.csv", index=False)
    exposure_b.to_csv(OUT / "batting_pressure_exposure_summary.csv", index=False)
    exposure_w.to_csv(OUT / "bowling_pressure_exposure_summary.csv", index=False)
    state_summary.to_csv(OUT / "pressure_diagnostics_validation.csv", index=False)

    print("\nPRESSURE INDEX DIAGNOSTICS COMPLETE")
    print(f"Project root: {ROOT}")
    print(f"Outputs: {OUT}")

    print("\nBATTING PRESSURE EXPOSURE")
    print(exposure_b.to_string(index=False))

    print("\nBOWLING PRESSURE EXPOSURE")
    print(exposure_w.to_string(index=False))

    print("\nVALIDATION")
    print(state_summary.to_string(index=False))

    print(
        "\nNo final Pressure Performance Index, component weights, "
        "or IPL/T20I combined score has been produced."
    )


if __name__ == "__main__":
    main()
