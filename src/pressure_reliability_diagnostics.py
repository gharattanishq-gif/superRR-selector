#!/usr/bin/env python3
"""Pressure-index reliability and shrinkage sensitivity diagnostics.

Reads the integrated batting/bowling pressure component tables and tests
empirical-Bayes-style shrinkage of component percentile scores toward 50.
This is a diagnostic exercise only: it does not select final weights or
publish final player rankings.

Run from project root:
    python src/pressure_reliability_diagnostics.py

Inputs:
    outputs/analysis/integrated_pressure_diagnostics/

Outputs:
    outputs/analysis/pressure_reliability_diagnostics/
"""
from __future__ import annotations

from pathlib import Path
import itertools
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "outputs" / "analysis" / "integrated_pressure_diagnostics"
OUTPUT_DIR = ROOT / "outputs" / "analysis" / "pressure_reliability_diagnostics"
POP_KEYS = ["competition_group", "evaluation_period"]

# k is the prior-equivalent evidence volume: reliability = n / (n + k).
# Match and volume reliability are combined geometrically so neither can
# completely substitute for the other.
SHRINKAGE_SCENARIOS = {
    "unshrunk": {"volume_k": 0.0, "matches_k": 0.0, "dismissals_k": 0.0},
    "light": {"volume_k": 20.0, "matches_k": 2.0, "dismissals_k": 2.0},
    "medium": {"volume_k": 40.0, "matches_k": 4.0, "dismissals_k": 4.0},
    "strong": {"volume_k": 70.0, "matches_k": 6.0, "dismissals_k": 6.0},
}

WEIGHT_SCENARIOS = {
    "70_20_10": (0.70, 0.20, 0.10),
    "60_30_10": (0.60, 0.30, 0.10),
    "60_20_20": (0.60, 0.20, 0.20),
    "50_30_20": (0.50, 0.30, 0.20),
}


def require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def safe_reliability(volume: pd.Series, matches: pd.Series,
                     volume_k: float, matches_k: float) -> pd.Series:
    v = pd.to_numeric(volume, errors="coerce").clip(lower=0)
    m = pd.to_numeric(matches, errors="coerce").clip(lower=0)
    if volume_k == 0 and matches_k == 0:
        return pd.Series(np.where(v.notna() & m.notna(), 1.0, np.nan), index=v.index)
    rv = v / (v + volume_k)
    rm = m / (m + matches_k)
    out = np.sqrt(rv * rm)
    return out.where(v.notna() & m.notna())


def percentile_by_population(df: pd.DataFrame, metric: str,
                             higher_is_better: bool) -> pd.Series:
    x = pd.to_numeric(df[metric], errors="coerce")
    if not higher_is_better:
        x = -x
    return x.groupby([df[k] for k in POP_KEYS], dropna=False).rank(
        method="average", pct=True
    ) * 100.0


def shrink_score(score: pd.Series, reliability: pd.Series) -> pd.Series:
    # Neutral prior is the population midpoint (50 percentile points).
    return 50.0 + reliability * (score - 50.0)


def row_mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    # Do not silently reweight incomplete metric blocks: require every metric.
    return x.mean(axis=1).where(x.notna().all(axis=1))


def add_batting_evidence(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["_high_balls"] = num(d, "high_pressure_balls_support")
    d["_high_matches"] = num(d, "high_pressure_matches")
    d["_high_dismissals"] = num(d, "high_dismissals")
    d["_baseline_balls"] = (num(d, "reconstructed_balls_faced") - d["_high_balls"]).clip(lower=0)
    d["_baseline_dismissals"] = (num(d, "dismissals") - d["_high_dismissals"]).clip(lower=0)
    d["_matched_balls"] = pd.concat([d["_high_balls"], d["_baseline_balls"]], axis=1).min(axis=1)
    d["_matched_dismissals"] = pd.concat([d["_high_dismissals"], d["_baseline_dismissals"]], axis=1).min(axis=1)
    d["_consistency_matches"] = num(d, "high_pressure_match_count_for_consistency")
    return d


def add_bowling_evidence(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["_high_balls"] = num(d, "high_legal_balls")
    d["_baseline_balls"] = num(d, "baseline_legal_balls")
    d["_matched_balls"] = pd.concat([d["_high_balls"], d["_baseline_balls"]], axis=1).min(axis=1)
    d["_high_matches"] = num(d, "high_pressure_matches")
    d["_consistency_matches"] = num(d, "high_pressure_match_count_for_consistency")
    d["_dot_uplift_support"] = num(d, "dot_uplift_weight")
    d["_dot_uplift_strata"] = num(d, "dot_uplift_comparable_strata")
    return d


def make_batting_components(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = add_batting_evidence(df)
    specs = {
        "abs_sr": ("high_pressure_strike_rate", True, "_high_balls", "_high_matches", "volume"),
        "abs_average": ("high_pressure_average", True, "_high_dismissals", "_high_matches", "dismissals"),
        "abs_boundary": ("high_pressure_boundary_run_pct", True, "_high_balls", "_high_matches", "volume"),
        "uplift_sr": ("pressure_sr_uplift", True, "_matched_balls", "_high_matches", "volume"),
        "uplift_average": ("pressure_average_uplift", True, "_matched_dismissals", "_high_matches", "dismissals"),
        "consistency": ("match_sr_mad_recalc", False, None, "_consistency_matches", "matches_only"),
    }
    for key, (source, direction, *_rest) in specs.items():
        d[key + "__raw_pct"] = percentile_by_population(d, source, direction)
    return d, list(specs)


def make_bowling_components(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = add_bowling_evidence(df)
    specs = {
        "abs_economy": ("high_pressure_economy", False, "_high_balls", "_high_matches", "volume"),
        "abs_wicket_rate": ("high_pressure_wickets_per_6_legal_balls", True, "_high_balls", "_high_matches", "volume"),
        "abs_dot_pct": ("high_pressure_dot_ball_pct", True, "_high_balls", "_high_matches", "volume"),
        "uplift_economy": ("pressure_economy_uplift", True, "_matched_balls", "_high_matches", "volume"),
        "uplift_wicket_rate": ("pressure_wicket_rate_uplift", True, "_matched_balls", "_high_matches", "volume"),
        "uplift_dot_pct": ("pressure_dot_ball_uplift", True, "_dot_uplift_support", "_high_matches", "volume"),
        "consistency": ("match_economy_mad_recalc", False, None, "_consistency_matches", "matches_only"),
    }
    for key, (source, direction, *_rest) in specs.items():
        d[key + "__raw_pct"] = percentile_by_population(d, source, direction)
    return d, list(specs)


def apply_reliability(df: pd.DataFrame, role: str, component_keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    detail_rows = []
    for scenario, pars in SHRINKAGE_SCENARIOS.items():
        for key in component_keys:
            raw_col = key + "__raw_pct"
            # Match component evidence definition to its metric family.
            if role == "Batting":
                if key == "abs_average":
                    volume_col, volume_k = "_high_dismissals", pars["dismissals_k"]
                elif key == "uplift_average":
                    volume_col, volume_k = "_matched_dismissals", pars["dismissals_k"]
                elif key in ("uplift_sr",):
                    volume_col, volume_k = "_matched_balls", pars["volume_k"]
                else:
                    volume_col, volume_k = "_high_balls", pars["volume_k"]
            else:
                if key == "uplift_dot_pct":
                    volume_col, volume_k = "_dot_uplift_support", pars["volume_k"]
                elif key.startswith("uplift_"):
                    volume_col, volume_k = "_matched_balls", pars["volume_k"]
                else:
                    volume_col, volume_k = "_high_balls", pars["volume_k"]

            match_col = "_consistency_matches" if key == "consistency" else "_high_matches"
            if key == "consistency":
                rel = safe_reliability(
                    pd.Series(1.0, index=d.index), d[match_col], 0.0, pars["matches_k"]
                )
            else:
                rel = safe_reliability(d[volume_col], d[match_col], volume_k, pars["matches_k"])
            # For dot-ball uplift, require at least one comparable phase/innings stratum.
            if role == "Bowling" and key == "uplift_dot_pct":
                rel = rel.where(d["_dot_uplift_strata"].fillna(0) >= 1)
            adj_col = f"{key}__{scenario}_pct"
            rel_col = f"{key}__{scenario}_reliability"
            d[rel_col] = rel
            d[adj_col] = shrink_score(d[raw_col], rel)
            for i in d.index:
                if pd.notna(d.at[i, raw_col]):
                    detail_rows.append({
                        "role": role,
                        "competition_group": d.at[i, "competition_group"],
                        "evaluation_period": d.at[i, "evaluation_period"],
                        "player_id": d.at[i, "striker_id"] if role == "Batting" else d.at[i, "bowler_id"],
                        "player_name": d.at[i, "striker_name"] if role == "Batting" else d.at[i, "bowler_name"],
                        "component": key,
                        "shrinkage_scenario": scenario,
                        "raw_percentile": d.at[i, raw_col],
                        "reliability": rel.at[i],
                        "adjusted_percentile": d.at[i, adj_col],
                        "evidence_volume": d.at[i, volume_col] if key != "consistency" else np.nan,
                        "high_pressure_matches": d.at[i, match_col],
                        "dot_uplift_comparable_strata": d.at[i, "_dot_uplift_strata"] if role == "Bowling" and key == "uplift_dot_pct" else np.nan,
                    })
    return d, pd.DataFrame(detail_rows)


def build_candidate_scores(df: pd.DataFrame, role: str, component_keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    if role == "Batting":
        blocks = {
            "absolute": ["abs_sr", "abs_average", "abs_boundary"],
            "uplift": ["uplift_sr", "uplift_average"],
            "consistency": ["consistency"],
        }
        id_col, name_col = "striker_id", "striker_name"
    else:
        blocks = {
            "absolute": ["abs_economy", "abs_wicket_rate", "abs_dot_pct"],
            "uplift": ["uplift_economy", "uplift_wicket_rate", "uplift_dot_pct"],
            "consistency": ["consistency"],
        }
        id_col, name_col = "bowler_id", "bowler_name"

    summary = []
    for shrink in SHRINKAGE_SCENARIOS:
        suffix = f"__{shrink}_pct"
        for block, keys in blocks.items():
            cols = [k + suffix for k in keys]
            d[f"{block}__{shrink}"] = row_mean(d, cols)
        for weight_name, weights in WEIGHT_SCENARIOS.items():
            score_col = f"PPI__{shrink}__{weight_name}"
            block_cols = [f"{b}__{shrink}" for b in ("absolute", "uplift", "consistency")]
            vals = d[block_cols]
            complete = vals.notna().all(axis=1)
            score = sum(vals.iloc[:, j] * weights[j] for j in range(3))
            d[score_col] = score.where(complete)
            for (comp, period), idx in d.groupby(POP_KEYS, dropna=False).groups.items():
                valid = d.loc[idx, score_col].dropna()
                summary.append({
                    "role": role, "competition_group": comp, "evaluation_period": period,
                    "shrinkage_scenario": shrink, "weight_scenario": weight_name,
                    "eligible_player_units": len(idx), "scored_player_units": len(valid),
                    "score_coverage_pct": 100 * len(valid) / len(idx) if len(idx) else np.nan,
                    "score_mean": valid.mean() if len(valid) else np.nan,
                    "score_median": valid.median() if len(valid) else np.nan,
                    "score_sd": valid.std(ddof=1) if len(valid) > 1 else np.nan,
                })
    return d, pd.DataFrame(summary)


def rank_comparisons(scores: pd.DataFrame, role: str) -> pd.DataFrame:
    id_col = "striker_id" if role == "Batting" else "bowler_id"
    cols = [c for c in scores.columns if c.startswith("PPI__")]
    rows = []
    for (comp, period), grp in scores.groupby(POP_KEYS, dropna=False):
        usable = [c for c in cols if grp[c].notna().sum() >= 10]
        if len(usable) < 2:
            continue
        ranks = pd.DataFrame(index=grp.index)
        for c in usable:
            ranks[c] = grp[c].rank(method="min", ascending=False)
        for a, b in itertools.combinations(usable, 2):
            pair = pd.concat([ranks[a], ranks[b]], axis=1).dropna()
            if pair.empty:
                continue
            top_a = set(grp.loc[grp[a].notna()].nlargest(10, a)[id_col].dropna())
            top_b = set(grp.loc[grp[b].notna()].nlargest(10, b)[id_col].dropna())
            diff = (pair.iloc[:, 0] - pair.iloc[:, 1]).abs()
            rows.append({
                "role": role, "competition_group": comp, "evaluation_period": period,
                "scenario_a": a, "scenario_b": b, "common_ranked_units": len(pair),
                "spearman_rank_correlation": pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"),
                "top10_overlap_count": len(top_a & top_b),
                "mean_absolute_rank_change": diff.mean(),
                "median_absolute_rank_change": diff.median(),
                "max_absolute_rank_change": diff.max(),
            })
    return pd.DataFrame(rows)


def validation_table(bat: pd.DataFrame, bowl: pd.DataFrame,
                     bat_scores: pd.DataFrame, bowl_scores: pd.DataFrame,
                     details: pd.DataFrame, ranks: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("eligible_batting_player_units", len(bat)),
        ("eligible_bowling_player_units", len(bowl)),
        ("batting_duplicate_population_player_keys", int(bat.duplicated(POP_KEYS + ["striker_id"]).sum())),
        ("bowling_duplicate_population_player_keys", int(bowl.duplicated(POP_KEYS + ["bowler_id"]).sum())),
        ("batting_rows_with_pressure_data", int(bat["high_pressure_matches"].notna().sum())),
        ("bowling_rows_with_pressure_data", int(bowl["high_pressure_matches"].notna().sum())),
        ("batting_candidate_score_columns", sum(c.startswith("PPI__") for c in bat_scores.columns)),
        ("bowling_candidate_score_columns", sum(c.startswith("PPI__") for c in bowl_scores.columns)),
        ("component_reliability_detail_rows", len(details)),
        ("rank_comparisons_generated", len(ranks)),
        ("scores_outside_0_100", int(((pd.concat([bat_scores.filter(regex=r"^PPI__"), bowl_scores.filter(regex=r"^PPI__")]).stack() < 0) | (pd.concat([bat_scores.filter(regex=r"^PPI__"), bowl_scores.filter(regex=r"^PPI__")]).stack() > 100)).sum())),
    ]
    return pd.DataFrame(rows, columns=["check", "value"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bat_path = INPUT_DIR / "integrated_batting_pressure_components.csv"
    bowl_path = INPUT_DIR / "integrated_bowling_pressure_components.csv"
    if not bat_path.exists() or not bowl_path.exists():
        raise FileNotFoundError(
            "Integrated pressure inputs not found. Run build_integrated_pressure_index_diagnostics.py first."
        )

    bat_raw = pd.read_csv(bat_path)
    bowl_raw = pd.read_csv(bowl_path)
    require_columns(bat_raw, [*POP_KEYS, "striker_id", "striker_name", "high_pressure_strike_rate",
        "high_pressure_average", "high_pressure_boundary_run_pct", "pressure_sr_uplift",
        "pressure_average_uplift", "match_sr_mad_recalc", "high_pressure_balls_support",
        "high_pressure_matches", "high_dismissals", "reconstructed_balls_faced", "dismissals",
        "high_pressure_match_count_for_consistency"], "batting input")
    require_columns(bowl_raw, [*POP_KEYS, "bowler_id", "bowler_name", "high_pressure_economy",
        "high_pressure_wickets_per_6_legal_balls", "high_pressure_dot_ball_pct",
        "pressure_economy_uplift", "pressure_wicket_rate_uplift", "pressure_dot_ball_uplift",
        "match_economy_mad_recalc", "high_legal_balls", "baseline_legal_balls",
        "high_pressure_matches", "high_pressure_match_count_for_consistency",
        "dot_uplift_weight", "dot_uplift_comparable_strata"], "bowling input")

    bat, bat_keys = make_batting_components(bat_raw)
    bowl, bowl_keys = make_bowling_components(bowl_raw)
    bat, bat_details = apply_reliability(bat, "Batting", bat_keys)
    bowl, bowl_details = apply_reliability(bowl, "Bowling", bowl_keys)
    bat_scores, bat_cov = build_candidate_scores(bat, "Batting", bat_keys)
    bowl_scores, bowl_cov = build_candidate_scores(bowl, "Bowling", bowl_keys)

    details = pd.concat([bat_details, bowl_details], ignore_index=True)
    coverage = pd.concat([bat_cov, bowl_cov], ignore_index=True)
    ranks = pd.concat([rank_comparisons(bat_scores, "Batting"), rank_comparisons(bowl_scores, "Bowling")], ignore_index=True)
    validation = validation_table(bat, bowl, bat_scores, bowl_scores, details, ranks)

    # Export enriched component tables, including raw and adjusted component percentiles
    # and candidate PPI scores for each shrinkage/weight combination.
    bat_scores.to_csv(OUTPUT_DIR / "batting_reliability_scores.csv", index=False)
    bowl_scores.to_csv(OUTPUT_DIR / "bowling_reliability_scores.csv", index=False)
    details.to_csv(OUTPUT_DIR / "component_reliability_detail.csv", index=False)
    coverage.to_csv(OUTPUT_DIR / "reliability_score_coverage_summary.csv", index=False)
    ranks.to_csv(OUTPUT_DIR / "reliability_rank_sensitivity.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "reliability_validation.csv", index=False)

    print("PRESSURE RELIABILITY DIAGNOSTICS COMPLETE")
    print(f"Output folder: {OUTPUT_DIR}")
    print("\nSHRINKAGE SCENARIOS")
    print(pd.DataFrame([
        {"scenario": name, **params} for name, params in SHRINKAGE_SCENARIOS.items()
    ]).to_string(index=False))
    print("\nSCORE COVERAGE (selected 70/20/10 scenario)")
    selected = coverage[coverage["weight_scenario"] == "70_20_10"]
    print(selected[["role", "competition_group", "evaluation_period", "shrinkage_scenario",
                    "eligible_player_units", "scored_player_units", "score_coverage_pct",
                    "score_median", "score_sd"]].to_string(index=False))
    print("\nVALIDATION")
    print(validation.to_string(index=False))
    print("\nNo final shrinkage strength, final weights, or final rankings were selected.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        main()
