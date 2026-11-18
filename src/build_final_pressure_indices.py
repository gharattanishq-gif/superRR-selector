from pathlib import Path
import numpy as np
import pandas as pd

"""
SupeRR Selector Season 2 — Round 1
Final Pressure Performance Index (PPI)

Primary decisions:
- Percentile normalisation within competition/evaluation period.
- Absolute / uplift / consistency blocks weighted 70% / 20% / 10%.
- Equal weighting within each block.
- No reliability shrinkage.
- Minimum two distinct high-pressure matches.
- Complete-case scoring: every defined component must be available.
- Missing pressure evidence is not treated as zero.

Run from the project root:
    python src\\build_final_pressure_indices.py
"""

# ---------------------------------------------------------------------------
# PATHS AND LOCKED METHODOLOGY
# ---------------------------------------------------------------------------
def find_root():
    here = Path(__file__).resolve()
    for candidate in [Path.cwd(), *here.parents]:
        if (candidate / "outputs" / "analysis" / "integrated_pressure_diagnostics").exists():
            return candidate
    raise FileNotFoundError(
        "Project root not found. Put this script in SupeRR/src and run it "
        "from the project root after integrated pressure diagnostics exist."
    )

ROOT = find_root()
INPUT_DIR = ROOT / "outputs" / "analysis" / "integrated_pressure_diagnostics"
OUTPUT_DIR = ROOT / "outputs" / "analysis" / "final_pressure_indices"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUTS = {
    "batting": INPUT_DIR / "integrated_batting_pressure_components.csv",
    "bowling": INPUT_DIR / "integrated_bowling_pressure_components.csv",
}

MIN_HIGH_PRESSURE_MATCHES = 2
ABSOLUTE_WEIGHT = 0.70
UPLIFT_WEIGHT = 0.20
CONSISTENCY_WEIGHT = 0.10

for label, path in INPUTS.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} component file: {path}")

# Each tuple: raw metric, score column, higher-is-better, block.
BATTING_METRICS = [
    ("high_pressure_strike_rate", "score_high_pressure_strike_rate", True, "absolute"),
    ("high_pressure_average", "score_high_pressure_average", True, "absolute"),
    ("high_pressure_boundary_run_pct", "score_high_pressure_boundary_run_pct", True, "absolute"),
    ("pressure_sr_uplift", "score_pressure_sr_uplift", True, "uplift"),
    ("pressure_average_uplift", "score_pressure_average_uplift", True, "uplift"),
    ("match_sr_mad_recalc", "score_match_sr_consistency", False, "consistency"),
]

BOWLING_METRICS = [
    ("high_pressure_economy", "score_high_pressure_economy", False, "absolute"),
    ("high_pressure_wickets_per_6_legal_balls", "score_high_pressure_wicket_rate", True, "absolute"),
    ("high_pressure_dot_ball_pct", "score_high_pressure_dot_ball_pct", True, "absolute"),
    ("pressure_economy_uplift", "score_pressure_economy_uplift", True, "uplift"),
    ("pressure_wicket_rate_uplift", "score_pressure_wicket_rate_uplift", True, "uplift"),
    ("pressure_dot_ball_uplift", "score_pressure_dot_ball_uplift", True, "uplift"),
    ("match_economy_mad_recalc", "score_match_economy_consistency", False, "consistency"),
]

def as_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y", "t"}
    )

def percentile_score(series, higher_is_better):
    values = pd.to_numeric(series, errors="coerce")
    if not higher_is_better:
        values = -values
    return values.rank(method="average", pct=True) * 100.0

def prepare_role(role, path, id_col, name_col, metrics):
    df = pd.read_csv(path, low_memory=False)
    required = {
        "population", "evaluation_period", "competition_group",
        id_col, name_col, "matches_played", "eligible_by_matches",
        "high_pressure_matches"
    }
    required.update(metric[0] for metric in metrics)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{role}: required columns missing: {missing}")

    df["evaluation_period"] = df["evaluation_period"].astype(str)
    df["eligible_by_matches"] = as_bool(df["eligible_by_matches"])
    if not df["eligible_by_matches"].all():
        bad = int((~df["eligible_by_matches"]).sum())
        raise ValueError(
            f"{role}: integrated component file contains {bad} ineligible rows; "
            "expected eligible-player populations only."
        )

    df["high_pressure_matches"] = pd.to_numeric(
        df["high_pressure_matches"], errors="coerce"
    )
    df["high_pressure_matches"] = df["high_pressure_matches"].fillna(0)

    # Preserve the authoritative eligible population, but only score players
    # with the minimum distinct-match support and every required component.
    metric_cols = [m[0] for m in metrics]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["has_minimum_pressure_matches"] = (
        df["high_pressure_matches"] >= MIN_HIGH_PRESSURE_MATCHES
    )
    df["available_component_count"] = df[metric_cols].notna().sum(axis=1)
    df["required_component_count"] = len(metric_cols)
    df["all_components_available"] = df[metric_cols].notna().all(axis=1)
    df["ppi_scored"] = (
        df["has_minimum_pressure_matches"] & df["all_components_available"]
    )

    df["ppi_exclusion_reason"] = np.select(
        [
            ~df["has_minimum_pressure_matches"],
            df["has_minimum_pressure_matches"] & ~df["all_components_available"],
        ],
        [
            "Fewer than 2 distinct high-pressure matches",
            "One or more required PPI components unavailable",
        ],
        default="Scored"
    )

    # Percentiles are calculated only among players who satisfy the same
    # complete-case scoring rule, within their competition/evaluation group.
    score_cols = []
    for raw_col, score_col, higher, block in metrics:
        df[score_col] = np.nan
        eligible_mask = df["ppi_scored"]
        df.loc[eligible_mask, score_col] = (
            df.loc[eligible_mask]
              .groupby(["competition_group", "evaluation_period"])[raw_col]
              .transform(lambda x: percentile_score(x, higher))
        )
        score_cols.append(score_col)

    # Equal metric weights within each conceptual block.
    absolute_cols = [score for _, score, _, block in metrics if block == "absolute"]
    uplift_cols = [score for _, score, _, block in metrics if block == "uplift"]
    consistency_cols = [score for _, score, _, block in metrics if block == "consistency"]

    df["absolute_performance_score"] = np.nan
    df["pressure_uplift_score"] = np.nan
    df["consistency_score"] = np.nan
    df["ppi_score"] = np.nan

    scored_mask = df["ppi_scored"]
    df.loc[scored_mask, "absolute_performance_score"] = (
        df.loc[scored_mask, absolute_cols].mean(axis=1)
    )
    df.loc[scored_mask, "pressure_uplift_score"] = (
        df.loc[scored_mask, uplift_cols].mean(axis=1)
    )
    df.loc[scored_mask, "consistency_score"] = (
        df.loc[scored_mask, consistency_cols].mean(axis=1)
    )
    df.loc[scored_mask, "ppi_score"] = (
        ABSOLUTE_WEIGHT * df.loc[scored_mask, "absolute_performance_score"]
        + UPLIFT_WEIGHT * df.loc[scored_mask, "pressure_uplift_score"]
        + CONSISTENCY_WEIGHT * df.loc[scored_mask, "consistency_score"]
    )

    # Rank only scored players. Rank 1 is best.
    df["ppi_rank"] = np.nan
    scored = df["ppi_scored"]
    df.loc[scored, "ppi_rank"] = (
        df.loc[scored]
          .groupby(["competition_group", "evaluation_period"])["ppi_score"]
          .rank(method="min", ascending=False)
    )
    df["top_10"] = scored & df["ppi_rank"].le(10)

    df["ppi_absolute_weight"] = ABSOLUTE_WEIGHT
    df["ppi_uplift_weight"] = UPLIFT_WEIGHT
    df["ppi_consistency_weight"] = CONSISTENCY_WEIGHT
    df["reliability_shrinkage_applied"] = False

    # Add clear support/evidence fields without changing the score.
    support_candidates = (
        ["high_pressure_matches", "high_pressure_balls_support"]
        if role == "Batting"
        else ["high_pressure_matches", "high_legal_balls"]
    )
    for col in support_candidates:
        if col not in df.columns:
            df[col] = np.nan

    df["role"] = role
    df["player_id"] = df[id_col]
    df["player_name"] = df[name_col]

    # Duplicate player units are a hard stop: ranks would otherwise be invalid.
    key = ["competition_group", "evaluation_period", id_col]
    duplicates = int(df.duplicated(key).sum())
    if duplicates:
        raise ValueError(f"{role}: found {duplicates} duplicate player units on {key}")

    # Scored values must remain on the 0–100 scale.
    scored_values = df.loc[scored, [
        "absolute_performance_score", "pressure_uplift_score",
        "consistency_score", "ppi_score"
    ]]
    if scored_values.isna().any().any():
        raise ValueError(f"{role}: scored rows contain missing block or PPI scores")
    if ((scored_values < -1e-9) | (scored_values > 100 + 1e-9)).any().any():
        raise ValueError(f"{role}: one or more scores fall outside 0–100")

    return df, score_cols, duplicates

bat, bat_score_cols, bat_duplicates = prepare_role(
    "Batting", INPUTS["batting"], "striker_id", "striker_name", BATTING_METRICS
)
bowl, bowl_score_cols, bowl_duplicates = prepare_role(
    "Bowling", INPUTS["bowling"], "bowler_id", "bowler_name", BOWLING_METRICS
)

# ---------------------------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------------------------
bat.to_csv(OUTPUT_DIR / "player_batting_pressure_index.csv", index=False)
bowl.to_csv(OUTPUT_DIR / "player_bowling_pressure_index.csv", index=False)

component_frames = []
for df, metrics in [(bat, BATTING_METRICS), (bowl, BOWLING_METRICS)]:
    id_fields = [
        "role", "competition_group", "evaluation_period",
        "player_id", "player_name", "matches_played",
        "high_pressure_matches", "ppi_scored", "ppi_exclusion_reason",
    ]
    for col in ["high_pressure_balls_support", "high_legal_balls"]:
        if col in df.columns:
            id_fields.append(col)
    id_fields += [
        "absolute_performance_score", "pressure_uplift_score",
        "consistency_score", "ppi_score", "ppi_rank", "top_10",
    ]
    out = df[id_fields].copy()
    for raw_col, score_col, _, block in metrics:
        out[raw_col] = df[raw_col]
        out[score_col] = df[score_col]
    component_frames.append(out)

components = pd.concat(component_frames, ignore_index=True, sort=False)
components.to_csv(OUTPUT_DIR / "pressure_index_component_scores.csv", index=False)

rankings = pd.concat([bat, bowl], ignore_index=True, sort=False)
rankings = rankings.loc[rankings["ppi_scored"]].copy()
rankings = rankings.sort_values(
    ["role", "competition_group", "evaluation_period", "ppi_rank", "player_name"]
)
rankings.to_csv(OUTPUT_DIR / "pressure_index_rankings.csv", index=False)

# Population-level validation and scoring coverage.
validation_rows = []
for role, df, duplicates in [
    ("Batting", bat, bat_duplicates),
    ("Bowling", bowl, bowl_duplicates),
]:
    for (comp, period), group in df.groupby(
        ["competition_group", "evaluation_period"], dropna=False
    ):
        eligible = len(group)
        scored_count = int(group["ppi_scored"].sum())
        validation_rows.append({
            "role": role,
            "competition_group": comp,
            "evaluation_period": period,
            "eligible_player_units": eligible,
            "scored_player_units": scored_count,
            "unscored_player_units": eligible - scored_count,
            "scoring_coverage_pct": (
                100.0 * scored_count / eligible if eligible else np.nan
            ),
            "players_below_minimum_matches": int(
                (~group["has_minimum_pressure_matches"]).sum()
            ),
            "players_missing_components_after_match_rule": int(
                (
                    group["has_minimum_pressure_matches"]
                    & ~group["all_components_available"]
                ).sum()
            ),
            "median_high_pressure_matches": group["high_pressure_matches"].median(),
            "median_high_pressure_balls_or_legal_balls": (
                lambda support: support.median() if support.notna().any() else np.nan
            )(
                pd.to_numeric(
                    group[
                        "high_pressure_balls_support"
                        if role == "Batting" else "high_legal_balls"
                    ], errors="coerce"
                )
            ),
            "duplicate_player_units": duplicates,
            "scores_outside_0_100": int(
                (
                    group.loc[group["ppi_scored"], "ppi_score"].lt(0)
                    | group.loc[group["ppi_scored"], "ppi_score"].gt(100)
                ).sum()
            ),
        })

validation = pd.DataFrame(validation_rows)
validation.to_csv(OUTPUT_DIR / "pressure_index_validation.csv", index=False)

print("FINAL PRESSURE PERFORMANCE INDEX BUILD COMPLETE")
print(f"Output folder: {OUTPUT_DIR}")
print("Methodology: percentile | 70/20/10 | equal weights within blocks")
print(f"Minimum distinct high-pressure matches: {MIN_HIGH_PRESSURE_MATCHES}")
print("Reliability shrinkage: not applied")
print("\nSCORING COVERAGE")
print(validation.to_string(index=False))
print("\nTOP 10 PREVIEW")
preview_cols = [
    "role", "competition_group", "evaluation_period",
    "ppi_rank", "player_name", "ppi_score",
    "high_pressure_matches", "ppi_exclusion_reason"
]
print(
    rankings.loc[rankings["ppi_rank"].le(10), preview_cols]
    .to_string(index=False)
)
print("\nVALIDATION")
print(f"Batting duplicate player units: {bat_duplicates}")
print(f"Bowling duplicate player units: {bowl_duplicates}")
print(f"Scored batting units: {int(bat['ppi_scored'].sum())}/{len(bat)}")
print(f"Scored bowling units: {int(bowl['ppi_scored'].sum())}/{len(bowl)}")
print("All scored component and PPI values are within 0–100.")
