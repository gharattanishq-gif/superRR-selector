from pathlib import Path
import math
import numpy as np
import pandas as pd

# SupeRR Selector Season 2 — Round 1
# Diagnostic normalisation comparison: percentile vs robust-z/CDF.
# This script creates PRELIMINARY equal-weight indices only.
# It does not finalise metric weights or the IPL/T20I combined score.

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
OUT = ROOT / "outputs" / "analysis" / "normalisation_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

BAT_PATH = IN / "player_batting_metrics.csv"
BOWL_PATH = IN / "player_bowling_metrics.csv"

MIN_BATTING_SR_COVERAGE = 90.0
MIN_BOWLING_DELIVERY_COVERAGE = 98.0
MIN_METRICS_FOR_INDEX = 3

BAT_METRICS = {
    "runs_per_innings": {"direction": 1, "availability": "runs_per_innings_complete"},
    "strike_rate": {"direction": 1, "availability": "strike_rate_available"},
    "batting_average": {"direction": 1, "availability": "batting_average_complete"},
    "boundary_runs_pct": {"direction": 1, "availability": "boundary_runs_pct_complete"},
}
BOWL_METRICS = {
    "wickets_per_innings": {"direction": 1, "availability": "wickets_per_innings_available"},
    "dot_ball_pct": {"direction": 1, "availability": "dot_ball_pct_available"},
    "economy": {"direction": -1, "availability": "economy_available"},
    "bowling_average": {"direction": -1, "availability": "bowling_average_available"},
}

def robust_score(series, direction):
    """Median/MAD robust z-score mapped through the normal CDF to 0–100."""
    x = pd.to_numeric(series, errors="coerce") * direction
    med = x.median()
    mad = (x - med).abs().median()
    if pd.isna(mad) or mad == 0:
        # Fall back to IQR scale if MAD is zero; if both collapse, use tied mid-score.
        q1, q3 = x.quantile([0.25, 0.75])
        scale = (q3 - q1) / 1.349
    else:
        scale = 1.4826 * mad
    if pd.isna(scale) or scale == 0:
        return pd.Series(np.where(x.notna(), 50.0, np.nan), index=series.index)
    z = ((x - med) / scale).clip(-3, 3)
    return z.map(lambda v: 50.0 * (1.0 + math.erf(v / math.sqrt(2))) if pd.notna(v) else np.nan)

def percentile_score(series, direction):
    x = pd.to_numeric(series, errors="coerce") * direction
    # Average ranks give tied values the same percentile score.
    return x.rank(method="average", pct=True) * 100.0

def spearman(a, b):
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    return pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")

def process_role(path, role, metrics, name_col, coverage_col=None, coverage_min=None):
    df = pd.read_csv(path, low_memory=False)
    required = {"population", "evaluation_period", "eligible_by_matches", name_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{role}: missing required columns: {sorted(missing)}")
    for metric, spec in metrics.items():
        for col in [metric, spec["availability"]]:
            if col not in df.columns:
                raise ValueError(f"{role}: missing required column {col}")

    eligible = df[df["eligible_by_matches"].astype(bool)].copy()
    eligible["entity_name"] = eligible[name_col]
    eligible["entity_id"] = eligible["striker_id" if role == "Batting" else "bowler_id"]

    if role == "Batting":
        eligible.loc[
            pd.to_numeric(eligible["strike_rate_coverage_pct"], errors="coerce")
            < MIN_BATTING_SR_COVERAGE,
            "strike_rate_available"
        ] = False
    if role == "Bowling":
        eligible.loc[
            pd.to_numeric(eligible["valid_delivery_coverage_pct"], errors="coerce")
            < MIN_BOWLING_DELIVERY_COVERAGE,
            list(metrics.keys())
        ] = np.nan

    output_parts = []
    summary_rows = []

    group_cols = ["population", "evaluation_period"]
    for (population, period), group in eligible.groupby(group_cols, dropna=False):
        g = group.copy()
        score_cols = []

        for metric, spec in metrics.items():
            available = g[spec["availability"]].fillna(False).astype(bool)
            values = pd.to_numeric(g[metric], errors="coerce")
            values = values.where(available)
            p_col = f"{metric}_percentile"
            r_col = f"{metric}_robust"
            g[p_col] = percentile_score(values, spec["direction"])
            g[r_col] = robust_score(values, spec["direction"])
            score_cols.append((p_col, r_col))

            summary_rows.append({
                "role": role,
                "population": population,
                "evaluation_period": period,
                "metric": metric,
                "eligible_players": len(g),
                "available_players": int(values.notna().sum()),
                "available_pct": round(100 * values.notna().mean(), 2),
                "raw_median": values.median(),
                "raw_p05": values.quantile(.05),
                "raw_p95": values.quantile(.95),
                "percentile_score_median": g[p_col].median(),
                "robust_score_median": g[r_col].median(),
            })

        p_cols = [x[0] for x in score_cols]
        r_cols = [x[1] for x in score_cols]
        g["metrics_available"] = g[p_cols].notna().sum(axis=1)
        g["index_percentile"] = g[p_cols].mean(axis=1).where(
            g["metrics_available"] >= MIN_METRICS_FOR_INDEX
        )
        g["index_robust"] = g[r_cols].mean(axis=1).where(
            g["metrics_available"] >= MIN_METRICS_FOR_INDEX
        )
        g["rank_percentile"] = g["index_percentile"].rank(
            ascending=False, method="min"
        )
        g["rank_robust"] = g["index_robust"].rank(
            ascending=False, method="min"
        )
        g["rank_difference_robust_minus_percentile"] = (
            g["rank_robust"] - g["rank_percentile"]
        )

        p_top = set(g.nsmallest(10, "rank_percentile")["entity_id"])
        r_top = set(g.nsmallest(10, "rank_robust")["entity_id"])
        summary = {
            "role": role,
            "population": population,
            "evaluation_period": period,
            "eligible_players": len(g),
            "players_with_index": int(g["index_percentile"].notna().sum()),
            "players_missing_index": int(g["index_percentile"].isna().sum()),
            "mean_metrics_available": round(g["metrics_available"].mean(), 3),
            "index_spearman_correlation": spearman(
                g["index_percentile"], g["index_robust"]
            ),
            "top10_overlap_count": len(p_top & r_top),
            "top10_overlap_pct": round(10 * len(p_top & r_top), 1),
            "median_absolute_rank_change": g[
                "rank_difference_robust_minus_percentile"
            ].abs().median(),
        }
        summary_rows.append(summary)
        output_parts.append(g)

    players = pd.concat(output_parts, ignore_index=True) if output_parts else pd.DataFrame()
    metrics_summary = pd.DataFrame(summary_rows)
    return players, metrics_summary

def main():
    batting, bat_summary = process_role(
        BAT_PATH, "Batting", BAT_METRICS, "striker_name"
    )
    bowling, bowl_summary = process_role(
        BOWL_PATH, "Bowling", BOWL_METRICS, "bowler_name"
    )

    batting.to_csv(OUT / "batting_normalisation_comparison.csv", index=False)
    bowling.to_csv(OUT / "bowling_normalisation_comparison.csv", index=False)

    bat_metric_summary = bat_summary[bat_summary["metric"].notna()]
    bowl_metric_summary = bowl_summary[bowl_summary["metric"].notna()]
    index_summary = pd.concat([
        bat_summary[bat_summary["metric"].isna()],
        bowl_summary[bowl_summary["metric"].isna()]
    ], ignore_index=True)

    pd.concat([bat_metric_summary, bowl_metric_summary], ignore_index=True).to_csv(
        OUT / "metric_availability_summary.csv", index=False
    )
    index_summary.to_csv(OUT / "normalisation_method_comparison.csv", index=False)

    print("\nNORMALISATION DIAGNOSTICS COMPLETE")
    print(f"Project root: {ROOT}")
    print(f"Outputs: {OUT}")
    print("\nINDEX METHOD COMPARISON")
    cols = [
        "role", "population", "evaluation_period", "eligible_players",
        "players_with_index", "players_missing_index",
        "index_spearman_correlation", "top10_overlap_count",
        "median_absolute_rank_change"
    ]
    print(index_summary[cols].to_string(index=False))
    print("\nMETRIC AVAILABILITY")
    print(pd.concat([bat_metric_summary, bowl_metric_summary], ignore_index=True)[
        ["role", "population", "evaluation_period", "metric",
         "eligible_players", "available_players", "available_pct"]
    ].to_string(index=False))
    print("\nNote: equal-weight indices are diagnostic only; no final weights are assigned.")
    print("No IPL/T20I combined scores are calculated in this stage.")

if __name__ == "__main__":
    main()
