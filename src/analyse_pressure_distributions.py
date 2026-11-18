from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "outputs" / "processed" / "pressure_states.csv"
OUTPUT_DIR = ROOT / "outputs" / "analysis" / "pressure_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DISTRIBUTION_FILE = OUTPUT_DIR / "pressure_distribution_summary.csv"
THRESHOLD_FILE = OUTPUT_DIR / "pressure_threshold_comparison.csv"
MATCH_FILE = OUTPUT_DIR / "match_pressure_summary.csv"
COVERAGE_FILE = OUTPUT_DIR / "pressure_calibration_coverage.csv"
PHASE_THRESHOLD_FILE = OUTPUT_DIR / "pressure_phase_threshold_comparison.csv"
PHASE_SEASON_FILE = OUTPUT_DIR / "pressure_phase_season_stability.csv"

# Candidate percentile definitions to compare.
THRESHOLD_SCHEMES = {
    "40_75": (40, 75),
    "30_70": (30, 70),
    "35_80": (35, 80),
}

# Minimum distinct matches before reporting a subgroup threshold.
MIN_MATCHES_FOR_SUBGROUP = 20

# Phase boundaries are based on each innings' scheduled legal-ball limit.
# For a standard 120-ball innings this reproduces:
# Powerplay = first 30 balls; Middle = balls 31-90; Death = final 30.
# For shortened innings, the same 25% / 50% / 25% structure is scaled
# to the actual scheduled innings length rather than assuming 120 balls.
def innings_phase(innings, balls_remaining, innings_max_balls):
    if (
        pd.isna(balls_remaining)
        or pd.isna(innings_max_balls)
        or float(innings_max_balls) <= 0
    ):
        return "Unknown"

    balls_remaining = float(balls_remaining)
    innings_max_balls = float(innings_max_balls)
    balls_bowled = innings_max_balls - balls_remaining

    powerplay_end = int(np.ceil(innings_max_balls * 0.25))
    middle_end = int(np.ceil(innings_max_balls * 0.75))

    if balls_bowled < powerplay_end:
        return "Powerplay"
    if balls_bowled < middle_end:
        return "Middle"
    return "Death"


# ============================================================
# 2. HELPERS
# ============================================================

def weighted_quantile(values, quantile, weights):
    """Weighted quantile using a stable, sorted cumulative-weight method."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    values = values[valid]
    weights = weights[valid]

    if len(values) == 0:
        return np.nan

    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)
    cutoff = float(quantile) * cumulative[-1]

    index = np.searchsorted(cumulative, cutoff, side="left")
    index = min(index, len(values) - 1)

    return float(values[index])


def distribution_stats(frame, value_col, weight_col):
    values = pd.to_numeric(frame[value_col], errors="coerce")
    weights = pd.to_numeric(frame[weight_col], errors="coerce")

    valid = values.notna() & weights.notna() & (weights > 0)
    values = values[valid]
    weights = weights[valid]

    if len(values) == 0:
        return {
            "rows": 0,
            "matches": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p30": np.nan,
            "p35": np.nan,
            "p40": np.nan,
            "median": np.nan,
            "p70": np.nan,
            "p75": np.nan,
            "p80": np.nan,
            "p90": np.nan,
            "max": np.nan,
        }

    total_weight = weights.sum()
    weighted_mean = float(np.average(values, weights=weights))
    weighted_variance = float(
        np.average((values - weighted_mean) ** 2, weights=weights)
    )

    return {
        "rows": int(len(values)),
        "matches": int(frame.loc[valid, "match_id"].nunique()),
        "mean": weighted_mean,
        "std": float(np.sqrt(weighted_variance)),
        "min": float(values.min()),
        "p10": weighted_quantile(values, 0.10, weights),
        "p25": weighted_quantile(values, 0.25, weights),
        "p30": weighted_quantile(values, 0.30, weights),
        "p35": weighted_quantile(values, 0.35, weights),
        "p40": weighted_quantile(values, 0.40, weights),
        "median": weighted_quantile(values, 0.50, weights),
        "p70": weighted_quantile(values, 0.70, weights),
        "p75": weighted_quantile(values, 0.75, weights),
        "p80": weighted_quantile(values, 0.80, weights),
        "p90": weighted_quantile(values, 0.90, weights),
        "max": float(values.max()),
    }


def make_group_label(columns, values):
    return " | ".join(
        f"{column}={value}"
        for column, value in zip(columns, values)
    )


# ============================================================
# 3. LOAD AND VALIDATE
# ============================================================

print("\nLoading pressure-state dataset...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {INPUT_FILE}. "
        "Run build_pressure_states.py first."
    )

data = pd.read_csv(INPUT_FILE, low_memory=False)

required = [
    "match_id",
    "match_date",
    "competition_group",
    "season",
    "innings",
    "legal_balls_remaining",
    "batting_pressure",
    "bowling_pressure",
    "pressure_eligible",
    "pressure_method",
    "run_rate_gap",
]

missing = [column for column in required if column not in data.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

data["match_date"] = pd.to_datetime(data["match_date"], errors="coerce")
data["innings"] = pd.to_numeric(data["innings"], errors="coerce")
data["legal_balls_remaining"] = pd.to_numeric(
    data["legal_balls_remaining"], errors="coerce"
)
data["batting_pressure"] = pd.to_numeric(
    data["batting_pressure"], errors="coerce"
)
data["bowling_pressure"] = pd.to_numeric(
    data["bowling_pressure"], errors="coerce"
)

if data["pressure_eligible"].dtype != bool:
    data["pressure_eligible"] = (
        data["pressure_eligible"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

eligible = data[
    data["pressure_eligible"]
    & data["batting_pressure"].notna()
    & data["bowling_pressure"].notna()
    & data["match_id"].notna()
    & data["competition_group"].notna()
].copy()

if eligible.empty:
    raise ValueError("No eligible pressure rows found.")

second_eligible = eligible[eligible["innings"].eq(2)].copy()
if second_eligible.empty:
    raise ValueError("No eligible second-innings pressure rows found.")
model_c_mask = second_eligible["pressure_method"].astype(str).str.contains(
    "Model C", case=False, na=False
)
if not model_c_mask.all():
    raise ValueError(
        "Eligible second-innings rows do not consistently use Model C. "
        "Re-run build_pressure_states.py before calibration."
    )
if second_eligible["run_rate_gap"].isna().all():
    raise ValueError("Model C run_rate_gap is entirely missing.")

eligible["innings"] = eligible["innings"].astype(int)
eligible["season"] = eligible["season"].astype(str)

# The maximum pre-delivery balls-remaining value identifies the scheduled
# legal-ball limit for that innings, including shortened second innings.
eligible["innings_max_balls"] = eligible.groupby(
    ["innings", "match_id"]
)["legal_balls_remaining"].transform("max")

eligible["phase"] = [
    innings_phase(innings, balls_remaining, max_balls)
    for innings, balls_remaining, max_balls in zip(
        eligible["innings"],
        eligible["legal_balls_remaining"],
        eligible["innings_max_balls"],
    )
]

# Equal-match weighting:
# - pooled thresholds: each match contributes total weight 1 across its
#   eligible rows in that innings and competition;
# - phase thresholds: each match contributes total weight 1 across its
#   eligible rows in that phase.
# Thus both distributions are match-weighted, not delivery-count-weighted.
eligible["match_row_count"] = eligible.groupby(
    ["innings", "competition_group", "match_id"]
)["match_id"].transform("size")
eligible["match_weight"] = 1.0 / eligible["match_row_count"]

print("Total source rows:", len(data))
print("Eligible pressure rows:", len(eligible))
print("Eligible matches:", eligible["match_id"].nunique())


# ============================================================
# 4. DISTRIBUTION SUMMARIES
# ============================================================

print("\nSummarising pressure distributions...")

group_definitions = [
    ("overall", ["innings"]),
    ("competition", ["innings", "competition_group"]),
    ("season", ["innings", "competition_group", "season"]),
    ("phase", ["innings", "competition_group", "phase"]),
]

distribution_rows = []

for level, group_columns in group_definitions:
    for group_values, group in eligible.groupby(
        group_columns, dropna=False, sort=True
    ):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        group_label = make_group_label(group_columns, group_values)

        for role, value_col in [
            ("Batting", "batting_pressure"),
            ("Bowling", "bowling_pressure"),
        ]:
            stats = distribution_stats(group, value_col, "match_weight")
            distribution_rows.append({
                "level": level,
                "group": group_label,
                "innings": int(group["innings"].iloc[0]),
                "competition_group": (
                    group["competition_group"].iloc[0]
                    if "competition_group" in group
                    else "All"
                ),
                "season": (
                    group["season"].iloc[0]
                    if "season" in group
                    else "All"
                ),
                "phase": (
                    group["phase"].iloc[0]
                    if "phase" in group
                    else "All"
                ),
                "role": role,
                **stats,
            })

distribution = pd.DataFrame(distribution_rows)
distribution.to_csv(DISTRIBUTION_FILE, index=False)


# ============================================================
# 5. THRESHOLD COMPARISON
# ============================================================

print("\nComparing candidate percentile thresholds...")

threshold_rows = []

# Primary calibration groups: innings x competition.
# Phase/season results are diagnostics, not independently imposed thresholds.
primary_groups = eligible.groupby(
    ["innings", "competition_group"],
    dropna=False,
    sort=True,
)

for (innings, comp_group), group in primary_groups:
    match_count = group["match_id"].nunique()

    if match_count < MIN_MATCHES_FOR_SUBGROUP:
        continue

    for role, value_col in [
        ("Batting", "batting_pressure"),
        ("Bowling", "bowling_pressure"),
    ]:
        values = group[value_col]
        weights = group["match_weight"]

        for scheme, (low_pct, high_pct) in THRESHOLD_SCHEMES.items():
            low_threshold = weighted_quantile(
                values, low_pct / 100, weights
            )
            high_threshold = weighted_quantile(
                values, high_pct / 100, weights
            )

            low_mask = values < low_threshold
            moderate_mask = (
                (values >= low_threshold)
                & (values <= high_threshold)
            )
            high_mask = values > high_threshold

            threshold_rows.append({
                "innings": int(innings),
                "competition_group": comp_group,
                "role": role,
                "scheme": scheme,
                "low_percentile": low_pct,
                "high_percentile": high_pct,
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
                "matches": match_count,
                "rows": len(group),
                "low_rows": int(low_mask.sum()),
                "moderate_rows": int(moderate_mask.sum()),
                "high_rows": int(high_mask.sum()),
                "low_matches": int(
                    group.loc[low_mask, "match_id"].nunique()
                ),
                "moderate_matches": int(
                    group.loc[moderate_mask, "match_id"].nunique()
                ),
                "high_matches": int(
                    group.loc[high_mask, "match_id"].nunique()
                ),
            })

thresholds = pd.DataFrame(threshold_rows)
thresholds.to_csv(THRESHOLD_FILE, index=False)


# ============================================================
# 6. PHASE-WISE THRESHOLDS AND SEASON STABILITY
# ============================================================

print("\\nEvaluating phase-wise thresholds and seasonal stability...")

phase_threshold_rows = []
phase_season_rows = []

# Within each phase, each contributing match again has total weight 1.
eligible["phase_match_row_count"] = eligible.groupby(
    ["innings", "competition_group", "phase", "match_id"]
)["match_id"].transform("size")
eligible["phase_match_weight"] = 1.0 / eligible["phase_match_row_count"]

# Explicitly verify the weighting basis used for the pooled-vs-phase
# comparison. Each match must contribute total weight 1 in either view.
pooled_weight_check = eligible.groupby(
    ["innings", "competition_group", "match_id"]
)["match_weight"].sum()
phase_weight_check = eligible.groupby(
    ["innings", "competition_group", "phase", "match_id"]
)["phase_match_weight"].sum()

if not np.allclose(pooled_weight_check.to_numpy(), 1.0):
    raise ValueError("Pooled equal-match weighting check failed.")
if not np.allclose(phase_weight_check.to_numpy(), 1.0):
    raise ValueError("Phase equal-match weighting check failed.")

for (innings, comp_group, phase), phase_group in eligible.groupby(
    ["innings", "competition_group", "phase"],
    dropna=False,
    sort=True,
):
    phase_match_count = phase_group["match_id"].nunique()
    if phase_match_count < MIN_MATCHES_FOR_SUBGROUP:
        continue

    pooled_group = eligible[
        eligible["innings"].eq(innings)
        & eligible["competition_group"].eq(comp_group)
    ]

    for role, value_col in [
        ("Batting", "batting_pressure"),
        ("Bowling", "bowling_pressure"),
    ]:
        for scheme, (low_pct, high_pct) in THRESHOLD_SCHEMES.items():
            phase_low = weighted_quantile(
                phase_group[value_col], low_pct / 100,
                phase_group["phase_match_weight"]
            )
            phase_high = weighted_quantile(
                phase_group[value_col], high_pct / 100,
                phase_group["phase_match_weight"]
            )
            pooled_low = weighted_quantile(
                pooled_group[value_col], low_pct / 100,
                pooled_group["match_weight"]
            )
            pooled_high = weighted_quantile(
                pooled_group[value_col], high_pct / 100,
                pooled_group["match_weight"]
            )

            phase_threshold_rows.append({
                "innings": int(innings),
                "competition_group": comp_group,
                "phase": phase,
                "role": role,
                "scheme": scheme,
                "matches": phase_match_count,
                "rows": len(phase_group),
                "phase_low_threshold": phase_low,
                "phase_high_threshold": phase_high,
                "pooled_low_threshold": pooled_low,
                "pooled_high_threshold": pooled_high,
                "low_threshold_vs_pooled": phase_low - pooled_low,
                "high_threshold_vs_pooled": phase_high - pooled_high,
            })

            for season, season_group in phase_group.groupby(
                "season", dropna=False, sort=True
            ):
                season_match_count = season_group["match_id"].nunique()
                if season_match_count < MIN_MATCHES_FOR_SUBGROUP:
                    continue

                season_low = weighted_quantile(
                    season_group[value_col], low_pct / 100,
                    season_group["phase_match_weight"]
                )
                season_high = weighted_quantile(
                    season_group[value_col], high_pct / 100,
                    season_group["phase_match_weight"]
                )

                phase_season_rows.append({
                    "innings": int(innings),
                    "competition_group": comp_group,
                    "phase": phase,
                    "season": season,
                    "role": role,
                    "scheme": scheme,
                    "matches": season_match_count,
                    "rows": len(season_group),
                    "season_low_threshold": season_low,
                    "season_high_threshold": season_high,
                    "phase_pooled_low_threshold": phase_low,
                    "phase_pooled_high_threshold": phase_high,
                    "low_threshold_deviation": season_low - phase_low,
                    "high_threshold_deviation": season_high - phase_high,
                })

phase_thresholds = pd.DataFrame(phase_threshold_rows)
phase_season_stability = pd.DataFrame(phase_season_rows)
phase_thresholds.to_csv(PHASE_THRESHOLD_FILE, index=False)
phase_season_stability.to_csv(PHASE_SEASON_FILE, index=False)



# ============================================================
# 6. MATCH-LEVEL PRESSURE COVERAGE
# ============================================================

print("\nBuilding match-level pressure summaries...")

match_summary = (
    eligible.groupby(
        ["match_id", "match_date", "competition_group", "season", "innings"],
        dropna=False,
        as_index=False,
    )
    .agg(
        eligible_pressure_rows=("batting_pressure", "size"),
        batting_pressure_mean=("batting_pressure", "mean"),
        batting_pressure_median=("batting_pressure", "median"),
        batting_pressure_p90=("batting_pressure", lambda x: x.quantile(0.90)),
        bowling_pressure_mean=("bowling_pressure", "mean"),
        bowling_pressure_median=("bowling_pressure", "median"),
        bowling_pressure_p90=("bowling_pressure", lambda x: x.quantile(0.90)),
        first_state_balls_remaining=("legal_balls_remaining", "max"),
        last_state_balls_remaining=("legal_balls_remaining", "min"),
        scheduled_innings_balls=("innings_max_balls", "max"),
    )
)

match_summary.to_csv(MATCH_FILE, index=False)


# ============================================================
# 7. COVERAGE AND QUALITY CHECKS
# ============================================================

coverage_rows = []

for (innings, comp_group), group in eligible.groupby(
    ["innings", "competition_group"], dropna=False
):
    coverage_rows.append({
        "innings": int(innings),
        "competition_group": comp_group,
        "eligible_rows": len(group),
        "eligible_matches": group["match_id"].nunique(),
        "seasons": group["season"].nunique(),
        "mean_rows_per_match": (
            len(group) / group["match_id"].nunique()
        ),
        "missing_batting_pressure": int(
            group["batting_pressure"].isna().sum()
        ),
        "missing_bowling_pressure": int(
            group["bowling_pressure"].isna().sum()
        ),
        "negative_batting_pressure_rows": int(
            (group["batting_pressure"] < 0).sum()
        ),
        "negative_bowling_pressure_rows": int(
            (group["bowling_pressure"] < 0).sum()
        ),
    })

coverage = pd.DataFrame(coverage_rows)
coverage.to_csv(COVERAGE_FILE, index=False)

# Validate the different pressure scales used by each innings.
# First innings: batting + bowling pressure = 0 (signed par deviation).
# Second innings: batting + bowling pressure = 1 (complementary probabilities).
eligible["pressure_sum"] = (
    eligible["batting_pressure"] + eligible["bowling_pressure"]
)

first_rows = eligible[eligible["innings"] == 1]
second_rows = eligible[eligible["innings"] == 2]

first_innings_error = (
    first_rows["pressure_sum"]
).abs()

second_innings_error = (
    second_rows["pressure_sum"] - 1.0
).abs()

max_first_error = (
    float(first_innings_error.max())
    if len(first_innings_error) else 0.0
)

max_second_error = (
    float(second_innings_error.max())
    if len(second_innings_error) else 0.0
)

if max(max_first_error, max_second_error) > 1e-8:
    raise ValueError(
        "Pressure relationship validation failed. "
        f"First-innings max error={max_first_error}; "
        f"second-innings max error={max_second_error}"
    )


# ============================================================
# 8. PRINT RESULTS
# ============================================================

print("\n" + "=" * 78)
print("PRESSURE DISTRIBUTION AND THRESHOLD ANALYSIS COMPLETE")
print("=" * 78)

print("\nEligible coverage:")
print(coverage.to_string(index=False))

print("\nPrimary threshold comparison:")
if thresholds.empty:
    print("No competition subgroup met the minimum match requirement.")
else:
    print(
        thresholds[
            [
                "innings",
                "competition_group",
                "role",
                "scheme",
                "low_threshold",
                "high_threshold",
                "low_matches",
                "moderate_matches",
                "high_matches",
            ]
        ].to_string(index=False)
    )

print("\nDistribution summary:")
print(
    distribution[
        distribution["level"].isin(["overall", "competition"])
    ][
        [
            "level",
            "group",
            "role",
            "rows",
            "matches",
            "mean",
            "std",
            "p30",
            "p35",
            "p40",
            "median",
            "p70",
            "p75",
            "p80",
        ]
    ].to_string(index=False)
)

print("\nFirst-innings zero-sum validation error:", max_first_error)
print("Second-innings sum-to-one validation error:", max_second_error)

print("\nFiles saved:")
print("\\nPhase-wise 40/75 threshold comparison:")
if phase_thresholds.empty:
    print("No phase subgroup met the minimum match requirement.")
else:
    print(
        phase_thresholds[phase_thresholds["scheme"].eq("40_75")][
            [
                "innings", "competition_group", "phase", "role", "matches",
                "phase_low_threshold", "phase_high_threshold",
                "low_threshold_vs_pooled", "high_threshold_vs_pooled",
            ]
        ].to_string(index=False)
    )

print("\\nPhase-by-season 40/75 stability:")
if phase_season_stability.empty:
    print("No phase-season subgroup met the minimum match requirement.")
else:
    print(
        phase_season_stability[phase_season_stability["scheme"].eq("40_75")][
            [
                "innings", "competition_group", "phase", "season", "role",
                "matches", "low_threshold_deviation", "high_threshold_deviation",
            ]
        ].to_string(index=False)
    )

print(DISTRIBUTION_FILE)
print(THRESHOLD_FILE)
print(MATCH_FILE)
print(COVERAGE_FILE)
print(PHASE_THRESHOLD_FILE)
print(PHASE_SEASON_FILE)

print("\n" + "=" * 78)
