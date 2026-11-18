from pathlib import Path
import math
import numpy as np
import pandas as pd

# SupeRR Selector Season 2 — Round 1
# Overall batting and bowling indices: percentile primary, robust sensitivity.
# IPL seasons and senior men's T20I are scored independently.
# This stage does NOT calculate pressure indices or combined IPL/T20I scores.

MIN_BATTING_SR_COVERAGE = 90.0
MIN_BOWLING_DELIVERY_COVERAGE = 98.0

BAT_METRICS = [
    "runs_per_innings",
    "strike_rate",
    "batting_average",
    "boundary_runs_pct",
]
BOWL_METRICS = [
    "wickets_per_innings",
    "dot_ball_pct",
    "economy",
    "bowling_average",
]

# All weights below are exploratory scenarios, not final recommendations.
WEIGHT_SCENARIOS = {
    "Batting": {
        "equal": {
            "runs_per_innings": 0.25,
            "strike_rate": 0.25,
            "batting_average": 0.25,
            "boundary_runs_pct": 0.25,
        },
        "productivity_tilt": {
            "runs_per_innings": 0.35,
            "strike_rate": 0.25,
            "batting_average": 0.25,
            "boundary_runs_pct": 0.15,
        },
        "scoring_rate_tilt": {
            "runs_per_innings": 0.25,
            "strike_rate": 0.30,
            "batting_average": 0.20,
            "boundary_runs_pct": 0.25,
        },
    },
    "Bowling": {
        "equal": {
            "wickets_per_innings": 0.25,
            "dot_ball_pct": 0.25,
            "economy": 0.25,
            "bowling_average": 0.25,
        },
        "wicket_threat_tilt": {
            "wickets_per_innings": 0.35,
            "dot_ball_pct": 0.20,
            "economy": 0.20,
            "bowling_average": 0.25,
        },
        "run_containment_tilt": {
            "wickets_per_innings": 0.20,
            "dot_ball_pct": 0.30,
            "economy": 0.30,
            "bowling_average": 0.20,
        },
    },
}

DIRECTIONS = {
    "runs_per_innings": 1,
    "strike_rate": 1,
    "batting_average": 1,
    "boundary_runs_pct": 1,
    "wickets_per_innings": 1,
    "dot_ball_pct": 1,
    "economy": -1,
    "bowling_average": -1,
}


def find_root():
    here = Path(__file__).resolve()
    for candidate in [Path.cwd(), *here.parents]:
        if (candidate / "outputs" / "analysis" / "player_metrics").exists():
            return candidate
    raise FileNotFoundError(
        "Project root not found. Put this script in SupeRR/src and run it from the project."
    )


ROOT = find_root()
IN = ROOT / "outputs" / "analysis" / "player_metrics"
OUT = ROOT / "outputs" / "analysis" / "overall_indices"
OUT.mkdir(parents=True, exist_ok=True)

BAT_PATH = IN / "player_batting_metrics.csv"
BOWL_PATH = IN / "player_bowling_metrics.csv"


def as_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin(
        ["1", "true", "t", "yes", "y"]
    )


def robust_score(series, direction):
    """Median/MAD robust z-score mapped through normal CDF to 0–100."""
    x = pd.to_numeric(series, errors="coerce") * direction
    med = x.median()
    mad = (x - med).abs().median()
    if pd.isna(mad) or mad == 0:
        q1, q3 = x.quantile([0.25, 0.75])
        scale = (q3 - q1) / 1.349
    else:
        scale = 1.4826 * mad
    if pd.isna(scale) or scale == 0:
        return pd.Series(np.where(x.notna(), 50.0, np.nan), index=series.index)
    z = ((x - med) / scale).clip(-3, 3)
    return z.map(
        lambda v: 50.0 * (1.0 + math.erf(v / math.sqrt(2)))
        if pd.notna(v) else np.nan
    )


def percentile_score(series, direction):
    x = pd.to_numeric(series, errors="coerce") * direction
    return x.rank(method="average", pct=True) * 100.0


def spearman(a, b):
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    return pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")


def validate_weights(role, metrics):
    for scenario, weights in WEIGHT_SCENARIOS[role].items():
        if set(weights) != set(metrics):
            raise ValueError(f"{role}/{scenario}: weights do not match metrics")
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError(f"{role}/{scenario}: weights must sum to 1")


def process_role(path, role, metrics, id_col, name_col, coverage_col=None, coverage_min=None):
    df = pd.read_csv(path, low_memory=False)
    required = {"population", "evaluation_period", "eligible_by_matches", id_col, name_col, "matches_played", *metrics}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{role}: missing required columns: {sorted(missing)}")

    eligible = df[as_bool(df["eligible_by_matches"])].copy()
    eligible["entity_id"] = eligible[id_col]
    eligible["entity_name"] = eligible[name_col]

    # Preserve every eligible unit that fails a quality/completeness rule.
    eligible["exclusion_reason"] = ""
    if role == "Batting":
        if "strike_rate_coverage_pct" not in eligible.columns:
            raise ValueError("Batting: strike_rate_coverage_pct is missing")
        coverage = pd.to_numeric(eligible["strike_rate_coverage_pct"], errors="coerce")
        low_coverage = coverage < MIN_BATTING_SR_COVERAGE
        eligible.loc[low_coverage, "exclusion_reason"] = (
            f"Strike-rate coverage below {MIN_BATTING_SR_COVERAGE:.0f}%"
        )
    else:
        if coverage_col not in eligible.columns:
            raise ValueError(f"Bowling: {coverage_col} is missing")
        coverage = pd.to_numeric(eligible[coverage_col], errors="coerce")
        low_coverage = coverage < coverage_min
        eligible.loc[low_coverage, "exclusion_reason"] = (
            f"Valid-delivery coverage below {coverage_min:.0f}%"
        )

    for metric in metrics:
        eligible[metric] = pd.to_numeric(eligible[metric], errors="coerce")
    missing_metric = eligible[metrics].isna().any(axis=1)
    eligible.loc[
        missing_metric & eligible["exclusion_reason"].eq(""), "exclusion_reason"
    ] = "One or more required index metrics are unavailable"
    excluded = eligible.loc[eligible["exclusion_reason"].ne("")].copy()
    eligible = eligible.loc[eligible["exclusion_reason"].eq("")].copy()

    outputs, sensitivity = [], []
    for (population, period), group in eligible.groupby(["population", "evaluation_period"], dropna=False):
        g = group.copy()
        for metric in metrics:
            g[f"{metric}_percentile_score"] = percentile_score(g[metric], DIRECTIONS[metric])
            g[f"{metric}_robust_score"] = robust_score(g[metric], DIRECTIONS[metric])

        for scenario, weights in WEIGHT_SCENARIOS[role].items():
            p_score = sum(g[f"{m}_percentile_score"] * w for m, w in weights.items())
            r_score = sum(g[f"{m}_robust_score"] * w for m, w in weights.items())
            g[f"index_percentile_{scenario}"] = p_score
            g[f"index_robust_{scenario}"] = r_score
            g[f"rank_percentile_{scenario}"] = p_score.rank(ascending=False, method="min")
            g[f"rank_robust_{scenario}"] = r_score.rank(ascending=False, method="min")

        g["overall_index_percentile"] = g["index_percentile_equal"]
        g["overall_index_robust_sensitivity"] = g["index_robust_equal"]
        g["overall_rank"] = g["rank_percentile_equal"]
        g["overall_rank_robust_sensitivity"] = g["rank_robust_equal"]
        g["index_method"] = "Percentile; equal-weight diagnostic baseline"
        g["four_metric_complete"] = True
        g["quality_threshold_passed"] = True

        equal_rank = g["rank_percentile_equal"]
        top_equal = set(g.nsmallest(10, "rank_percentile_equal")["entity_id"])
        for scenario in WEIGHT_SCENARIOS[role]:
            scenario_rank = g[f"rank_percentile_{scenario}"]
            top_scenario = set(g.nsmallest(10, f"rank_percentile_{scenario}")["entity_id"])
            sensitivity.append({
                "role": role, "population": population, "evaluation_period": period,
                "scenario": scenario, "players_scored": len(g),
                "top10_overlap_vs_equal": len(top_equal & top_scenario),
                "rank_spearman_vs_equal": spearman(equal_rank, scenario_rank),
                "median_absolute_rank_change_vs_equal": (scenario_rank - equal_rank).abs().median(),
                "max_absolute_rank_change_vs_equal": (scenario_rank - equal_rank).abs().max(),
            })

        top_robust = set(g.nsmallest(10, "rank_robust_equal")["entity_id"])
        sensitivity.append({
            "role": role, "population": population, "evaluation_period": period,
            "scenario": "robust_normalisation_equal_weights", "players_scored": len(g),
            "top10_overlap_vs_equal": len(top_equal & top_robust),
            "rank_spearman_vs_equal": spearman(g["rank_percentile_equal"], g["rank_robust_equal"]),
            "median_absolute_rank_change_vs_equal": (g["rank_robust_equal"] - equal_rank).abs().median(),
            "max_absolute_rank_change_vs_equal": (g["rank_robust_equal"] - equal_rank).abs().max(),
        })
        outputs.append(g)

    scored = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    return scored, pd.DataFrame(sensitivity), excluded


def main():
    if not BAT_PATH.exists() or not BOWL_PATH.exists():
        raise FileNotFoundError("Expected player_batting_metrics.csv and player_bowling_metrics.csv in outputs/analysis/player_metrics")
    validate_weights("Batting", BAT_METRICS)
    validate_weights("Bowling", BOWL_METRICS)

    batting, bat_sens, bat_excluded = process_role(BAT_PATH, "Batting", BAT_METRICS, "striker_id", "striker_name")
    bowling, bowl_sens, bowl_excluded = process_role(BOWL_PATH, "Bowling", BOWL_METRICS, "bowler_id", "bowler_name", "valid_delivery_coverage_pct", MIN_BOWLING_DELIVERY_COVERAGE)

    batting.to_csv(OUT / "player_overall_batting_indices.csv", index=False)
    bowling.to_csv(OUT / "player_overall_bowling_indices.csv", index=False)
    bat_excluded.to_csv(OUT / "batters_excluded_from_primary_index.csv", index=False)
    bowl_excluded.to_csv(OUT / "bowlers_excluded_from_primary_index.csv", index=False)
    sensitivity = pd.concat([bat_sens, bowl_sens], ignore_index=True)
    sensitivity.to_csv(OUT / "index_weight_and_method_sensitivity.csv", index=False)

    weight_rows = []
    for role, scenarios in WEIGHT_SCENARIOS.items():
        for scenario, weights in scenarios.items():
            for metric, weight in weights.items():
                weight_rows.append({"role": role, "scenario": scenario, "metric": metric, "weight_pct": weight * 100, "status": "Exploratory only; not final"})
    pd.DataFrame(weight_rows).to_csv(OUT / "exploratory_weight_scenarios.csv", index=False)

    # Validate the expected populations and strict four-metric completeness.
    bat_source = pd.read_csv(BAT_PATH, low_memory=False)
    bowl_source = pd.read_csv(BOWL_PATH, low_memory=False)
    validation = pd.DataFrame([
        {"check": "Eligible batting units before quality filter", "value": int(as_bool(bat_source["eligible_by_matches"]).sum())},
        {"check": "Batting units scored (>=90% SR coverage; all four metrics)", "value": len(batting)},
        {"check": "Batting units excluded from primary index", "value": len(bat_excluded)},
        {"check": "Eligible bowling units before quality filter", "value": int(as_bool(bowl_source["eligible_by_matches"]).sum())},
        {"check": "Bowling units scored (>=98% delivery coverage; all four metrics)", "value": len(bowling)},
        {"check": "IPL/T20I combined scores created", "value": "No"},
        {"check": "Pressure index created", "value": "No"},
        {"check": "Final weights assigned", "value": "No; alternative scenarios are exploratory"},
    ])
    validation.to_csv(OUT / "overall_index_validation.csv", index=False)

    print("\nOVERALL INDEX BUILD COMPLETE")
    print(f"Project root: {ROOT}")
    print(f"Outputs: {OUT}")
    for role, frame in [("BATTING", batting), ("BOWLING", bowling)]:
        print(f"\n{role}: equal-weight percentile baseline")
        print(frame.groupby(["population", "evaluation_period"]).agg(players_scored=("entity_id", "nunique"), median_index=("overall_index_percentile", "median"), top_index=("overall_index_percentile", "max")).to_string())
    print("\nWEIGHT / METHOD SENSITIVITY")
    print(sensitivity.to_string(index=False))
    print("\nVALIDATION")
    print(validation.to_string(index=False))
    print("\nNo pressure index, final weights, or IPL/T20I combined score has been produced.")


if __name__ == "__main__":
    main()
