#!/usr/bin/env python3
"""Pressure-index component redundancy and weighting sensitivity diagnostics.

Reads the integrated pressure component tables created by
build_integrated_pressure_index_diagnostics.py and writes diagnostics only.
It does NOT choose final weights, apply reliability shrinkage, or publish rankings.

Expected project layout:
  outputs/analysis/integrated_pressure_diagnostics/

Run from project root:
  python src/pressure_index_sensitivity.py
"""
from __future__ import annotations

from pathlib import Path
import itertools
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "outputs" / "analysis" / "integrated_pressure_diagnostics"
OUTPUT_DIR = ROOT / "outputs" / "analysis" / "pressure_index_sensitivity"

POP_KEYS = ["competition_group", "evaluation_period"]
WEIGHT_SCENARIOS = {
    "70_20_10": (0.70, 0.20, 0.10),
    "60_30_10": (0.60, 0.30, 0.10),
    "60_20_20": (0.60, 0.20, 0.20),
    "50_30_20": (0.50, 0.30, 0.20),
}


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Within-population percentile score, 0-100; preserve missing values."""
    x = pd.to_numeric(series, errors="coerce")
    if not higher_is_better:
        x = -x
    out = x.rank(method="average", pct=True) * 100.0
    return out.where(x.notna())


def mean_available(df: pd.DataFrame, cols: list[str], min_count: int | None = None) -> pd.Series:
    vals = df[cols].apply(pd.to_numeric, errors="coerce")
    if min_count is None:
        min_count = len(cols)
    result = vals.mean(axis=1, skipna=True)
    return result.where(vals.notna().sum(axis=1) >= min_count)


def add_batting_scores(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Recompute percentiles consistently within each IPL season / combined T20I pool.
    metric_specs = {
        "high_pressure_strike_rate": True,
        "high_pressure_average": True,
        "high_pressure_boundary_run_pct": True,
        "pressure_sr_uplift": True,
        "pressure_average_uplift": True,
        "match_sr_mad_recalc": False,
    }
    for metric, direction in metric_specs.items():
        d[metric + "__pct"] = np.nan
    for _, idx in d.groupby(POP_KEYS, dropna=False).groups.items():
        for metric, direction in metric_specs.items():
            d.loc[idx, metric + "__pct"] = percentile(d.loc[idx, metric], direction)

    d["absolute_full"] = mean_available(d, [
        "high_pressure_strike_rate__pct",
        "high_pressure_average__pct",
        "high_pressure_boundary_run_pct__pct",
    ])
    d["uplift_full"] = mean_available(d, [
        "pressure_sr_uplift__pct", "pressure_average_uplift__pct"
    ])
    d["uplift_sr_only"] = d["pressure_sr_uplift__pct"]
    d["consistency"] = d["match_sr_mad_recalc__pct"]
    return d


def add_bowling_scores(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    metric_specs = {
        "high_pressure_economy": False,
        "high_pressure_bowling_average": False,
        "high_pressure_wickets_per_6_legal_balls": True,
        "high_pressure_dot_ball_pct": True,
        "pressure_economy_uplift": True,
        "pressure_wicket_rate_uplift": True,
        "pressure_dot_ball_uplift": True,
        "match_economy_mad_recalc": False,
    }
    for metric in metric_specs:
        d[metric + "__pct"] = np.nan
    for _, idx in d.groupby(POP_KEYS, dropna=False).groups.items():
        for metric, direction in metric_specs.items():
            d.loc[idx, metric + "__pct"] = percentile(d.loc[idx, metric], direction)

    d["absolute_avg_spec"] = mean_available(d, [
        "high_pressure_economy__pct",
        "high_pressure_bowling_average__pct",
        "high_pressure_dot_ball_pct__pct",
    ])
    d["absolute_wicket_spec"] = mean_available(d, [
        "high_pressure_economy__pct",
        "high_pressure_wickets_per_6_legal_balls__pct",
        "high_pressure_dot_ball_pct__pct",
    ])
    d["absolute_four_metric"] = mean_available(d, [
        "high_pressure_economy__pct",
        "high_pressure_bowling_average__pct",
        "high_pressure_wickets_per_6_legal_balls__pct",
        "high_pressure_dot_ball_pct__pct",
    ])
    d["uplift_full"] = mean_available(d, [
        "pressure_economy_uplift__pct",
        "pressure_wicket_rate_uplift__pct",
        "pressure_dot_ball_uplift__pct",
    ])
    d["uplift_no_dot"] = mean_available(d, [
        "pressure_economy_uplift__pct",
        "pressure_wicket_rate_uplift__pct",
    ])
    d["consistency"] = d["match_economy_mad_recalc__pct"]
    return d


def make_scenario_scores(df: pd.DataFrame, role: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    score_cols = []
    if role == "Batting":
        structures = {
            "BAT_full_uplift": ("absolute_full", "uplift_full", "consistency"),
            "BAT_SR_uplift_only": ("absolute_full", "uplift_sr_only", "consistency"),
        }
        id_col = "striker_id"
        name_col = "striker_name"
    else:
        structures = {
            "BOWL_average_spec": ("absolute_avg_spec", "uplift_full", "consistency"),
            "BOWL_wicket_rate_spec": ("absolute_wicket_spec", "uplift_full", "consistency"),
            "BOWL_four_metric_absolute": ("absolute_four_metric", "uplift_full", "consistency"),
            "BOWL_average_no_dot_uplift": ("absolute_avg_spec", "uplift_no_dot", "consistency"),
        }
        id_col = "bowler_id"
        name_col = "bowler_name"

    out = df[[*POP_KEYS, id_col, name_col]].copy()
    for structure, blocks in structures.items():
        for weight_name, weights in WEIGHT_SCENARIOS.items():
            score_name = f"{structure}__{weight_name}"
            block_df = df[list(blocks)].apply(pd.to_numeric, errors="coerce")
            # Strict complete-case score: all three conceptual blocks must be available.
            complete = block_df.notna().all(axis=1)
            score = sum(block_df.iloc[:, i] * weights[i] for i in range(3))
            out[score_name] = score.where(complete)
            score_cols.append(score_name)
            for (comp, period), gidx in out.groupby(POP_KEYS, dropna=False).groups.items():
                valid = out.loc[gidx, score_name].dropna()
                rows.append({
                    "role": role,
                    "competition_group": comp,
                    "evaluation_period": period,
                    "structure": structure,
                    "weight_scenario": weight_name,
                    "eligible_player_units": len(gidx),
                    "scored_player_units": int(valid.size),
                    "score_coverage_pct": (100 * valid.size / len(gidx)) if len(gidx) else np.nan,
                    "score_median": valid.median() if len(valid) else np.nan,
                    "score_mean": valid.mean() if len(valid) else np.nan,
                    "score_sd": valid.std(ddof=1) if len(valid) > 1 else np.nan,
                    "score_min": valid.min() if len(valid) else np.nan,
                    "score_max": valid.max() if len(valid) else np.nan,
                })
    return out, pd.DataFrame(rows)


def rank_sensitivity(scores: pd.DataFrame, role: str) -> pd.DataFrame:
    id_col = "striker_id" if role == "Batting" else "bowler_id"
    rows = []
    score_cols = [c for c in scores.columns if "__" in c and c not in POP_KEYS]
    for (comp, period), grp in scores.groupby(POP_KEYS, dropna=False):
        valid_cols = [c for c in score_cols if grp[c].notna().sum() >= 10]
        if len(valid_cols) < 2:
            continue
        ranks = grp[[id_col, *valid_cols]].copy()
        for c in valid_cols:
            ranks[c + "__rank"] = ranks[c].rank(method="min", ascending=False)
        rank_cols = [c + "__rank" for c in valid_cols]
        for a, b in itertools.combinations(valid_cols, 2):
            common = ranks[[a + "__rank", b + "__rank"]].dropna()
            if common.empty:
                continue
            top_a = set(ranks.nsmallest(10, a)[id_col].dropna())
            top_b = set(ranks.nsmallest(10, b)[id_col].dropna())
            overlap = len(top_a & top_b)
            rho = common[a + "__rank"].corr(common[b + "__rank"], method="spearman") if len(common) > 1 else np.nan
            diff = (common[a + "__rank"] - common[b + "__rank"]).abs()
            rows.append({
                "role": role,
                "competition_group": comp,
                "evaluation_period": period,
                "scenario_a": a,
                "scenario_b": b,
                "common_player_units": len(common),
                "spearman_rank_correlation": rho,
                "top10_overlap_count": overlap,
                "top10_overlap_pct_of_10": overlap * 10.0,
                "mean_absolute_rank_change": diff.mean(),
                "median_absolute_rank_change": diff.median(),
                "max_absolute_rank_change": diff.max(),
            })
    return pd.DataFrame(rows)


def correlation_outputs(bat: pd.DataFrame, bowl: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "Batting": [
            "high_pressure_strike_rate", "high_pressure_average",
            "high_pressure_boundary_run_pct", "pressure_sr_uplift",
            "pressure_average_uplift", "match_sr_mad_recalc",
        ],
        "Bowling": [
            "high_pressure_economy", "high_pressure_bowling_average",
            "high_pressure_wickets_per_6_legal_balls", "high_pressure_dot_ball_pct",
            "pressure_economy_uplift", "pressure_wicket_rate_uplift",
            "pressure_dot_ball_uplift", "match_economy_mad_recalc",
        ],
    }
    frames = []
    for role, df in [("Batting", bat), ("Bowling", bowl)]:
        for (comp, period), grp in df.groupby(POP_KEYS, dropna=False):
            cols = [c for c in specs[role] if c in grp.columns]
            corr = grp[cols].corr(method="spearman", min_periods=5)
            for a in cols:
                for b in cols:
                    if cols.index(b) < cols.index(a):
                        continue
                    pair = grp[[a, b]].dropna()
                    frames.append({
                        "role": role,
                        "competition_group": comp,
                        "evaluation_period": period,
                        "metric_a": a,
                        "metric_b": b,
                        "n_complete_pairs": len(pair),
                        "spearman_correlation": corr.loc[a, b] if len(pair) >= 5 else np.nan,
                    })
    return pd.DataFrame(frames)


def main() -> None:
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    batting_path = INPUT_DIR / "integrated_batting_pressure_components.csv"
    bowling_path = INPUT_DIR / "integrated_bowling_pressure_components.csv"
    if not batting_path.exists() or not bowling_path.exists():
        raise FileNotFoundError(
            "Integrated component files not found. Expected both files in "
            f"{INPUT_DIR}. Run build_integrated_pressure_index_diagnostics.py first."
        )

    batting = pd.read_csv(batting_path)
    bowling = pd.read_csv(bowling_path)
    require_columns(batting, POP_KEYS + ["striker_id", "striker_name", "high_pressure_strike_rate",
        "high_pressure_average", "high_pressure_boundary_run_pct", "pressure_sr_uplift",
        "pressure_average_uplift", "match_sr_mad_recalc"], "Batting components")
    require_columns(bowling, POP_KEYS + ["bowler_id", "bowler_name", "high_pressure_economy",
        "high_pressure_bowling_average", "high_pressure_wickets_per_6_legal_balls",
        "high_pressure_dot_ball_pct", "pressure_economy_uplift", "pressure_wicket_rate_uplift",
        "pressure_dot_ball_uplift", "match_economy_mad_recalc"], "Bowling components")

    batting = add_batting_scores(batting)
    bowling = add_bowling_scores(bowling)
    bat_scores, bat_summary = make_scenario_scores(batting, "Batting")
    bowl_scores, bowl_summary = make_scenario_scores(bowling, "Bowling")

    correlations = correlation_outputs(batting, bowling)
    coverage = pd.concat([bat_summary, bowl_summary], ignore_index=True)
    rank_sens = pd.concat([
        rank_sensitivity(bat_scores, "Batting"),
        rank_sensitivity(bowl_scores, "Bowling"),
    ], ignore_index=True)

    # Save player-level component percentiles and candidate scenario scores for audit.
    batting.to_csv(OUTPUT_DIR / "batting_component_percentiles.csv", index=False)
    bowling.to_csv(OUTPUT_DIR / "bowling_component_percentiles.csv", index=False)
    bat_scores.to_csv(OUTPUT_DIR / "batting_candidate_scenario_scores.csv", index=False)
    bowl_scores.to_csv(OUTPUT_DIR / "bowling_candidate_scenario_scores.csv", index=False)
    correlations.to_csv(OUTPUT_DIR / "pressure_component_correlations.csv", index=False)
    coverage.to_csv(OUTPUT_DIR / "candidate_score_coverage_summary.csv", index=False)
    rank_sens.to_csv(OUTPUT_DIR / "candidate_rank_sensitivity.csv", index=False)

    validation = pd.DataFrame([
        {"check": "batting_input_rows", "value": len(batting)},
        {"check": "bowling_input_rows", "value": len(bowling)},
        {"check": "batting_duplicate_population_player_keys", "value": int(batting.duplicated(POP_KEYS + ["striker_id"]).sum())},
        {"check": "bowling_duplicate_population_player_keys", "value": int(bowling.duplicated(POP_KEYS + ["bowler_id"]).sum())},
        {"check": "batting_candidate_score_columns", "value": len([c for c in bat_scores if c not in POP_KEYS + ["striker_id", "striker_name"]])},
        {"check": "bowling_candidate_score_columns", "value": len([c for c in bowl_scores if c not in POP_KEYS + ["bowler_id", "bowler_name"]])},
        {"check": "candidate_scenarios_with_at_least_10_scored_units", "value": int((coverage["scored_player_units"] >= 10).sum())},
        {"check": "rank_comparisons_generated", "value": len(rank_sens)},
    ])
    validation.to_csv(OUTPUT_DIR / "pressure_index_sensitivity_validation.csv", index=False)

    print("PRESSURE INDEX SENSITIVITY DIAGNOSTICS COMPLETE")
    print(f"Output folder: {OUTPUT_DIR}")
    print("\nCANDIDATE SCORE COVERAGE")
    print(coverage[["role", "competition_group", "evaluation_period", "structure",
                    "weight_scenario", "eligible_player_units", "scored_player_units",
                    "score_coverage_pct"]].to_string(index=False))
    print("\nVALIDATION")
    print(validation.to_string(index=False))
    print("\nNo final weights, reliability shrinkage, or final rankings were selected.")


if __name__ == "__main__":
    main()
