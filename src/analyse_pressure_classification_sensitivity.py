from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PRESSURE CLASSIFICATION SENSITIVITY CHECK
# ============================================================
# Compares pooled 40th/75th percentile thresholds against
# phase-specific 40th/75th thresholds on the SAME eligible rows.
#
# Main decision group: first-innings IPL.
# Other innings/competition groups are included as diagnostics.
#
# Equal-match weighting is used for both pooled and phase
# distributions. Within each distribution, every match contributes
# total weight 1, preventing longer/more densely represented matches
# from dominating the percentile cutoffs.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "outputs" / "processed" / "pressure_states.csv"
OUTPUT_DIR = ROOT / "outputs" / "analysis" / "pressure_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_OUTPUT = OUTPUT_DIR / "pressure_classification_sensitivity_thresholds.csv"
ROW_OUTPUT = OUTPUT_DIR / "pressure_classification_sensitivity_rows.csv"
MATCH_OUTPUT = OUTPUT_DIR / "pressure_classification_sensitivity_matches.csv"
SEASON_OUTPUT = OUTPUT_DIR / "pressure_classification_sensitivity_seasons.csv"

LOW_PERCENTILE = 0.40
HIGH_PERCENTILE = 0.75
MIN_MATCHES = 20

ROLE_COLUMNS = {
    "Batting": "batting_pressure",
    "Bowling": "bowling_pressure",
}


def weighted_quantile(values, quantile, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
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
    return float(values[min(index, len(values) - 1)])


def classify(values, low, high):
    # Preserve the exact boundary convention used in the calibration:
    # Low < low threshold; Moderate includes both thresholds;
    # High > high threshold.
    return np.where(
        values < low,
        "Low",
        np.where(values > high, "High", "Moderate"),
    )


def phase_from_balls(balls_remaining, max_balls):
    if pd.isna(balls_remaining) or pd.isna(max_balls) or max_balls <= 0:
        return "Unknown"

    balls_bowled = float(max_balls) - float(balls_remaining)
    powerplay_end = int(np.ceil(float(max_balls) * 0.25))
    middle_end = int(np.ceil(float(max_balls) * 0.75))

    if balls_bowled < powerplay_end:
        return "Powerplay"
    if balls_bowled < middle_end:
        return "Middle"
    return "Death"


def safe_pct(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else np.nan


def main():
    print("\\nLoading pressure-state dataset...")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. Run build_pressure_states.py first."
        )

    data = pd.read_csv(INPUT_FILE, low_memory=False)
    required = [
        "match_id", "competition_group", "season", "innings",
        "legal_balls_remaining", "batting_pressure", "bowling_pressure",
        "pressure_eligible", "pressure_method", "run_rate_gap",
    ]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in [
        "innings", "legal_balls_remaining",
        "batting_pressure", "bowling_pressure",
    ]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if data["pressure_eligible"].dtype != bool:
        data["pressure_eligible"] = (
            data["pressure_eligible"].astype(str).str.strip().str.lower()
            .isin(["true", "1", "yes"])
        )

    eligible = data[
        data["pressure_eligible"]
        & data["match_id"].notna()
        & data["competition_group"].notna()
        & data["innings"].notna()
        & data["batting_pressure"].notna()
        & data["bowling_pressure"].notna()
    ].copy()

    if eligible.empty:
        raise ValueError("No eligible pressure rows found.")

    eligible["innings"] = eligible["innings"].astype(int)
    eligible["season"] = eligible["season"].astype(str)

    second = eligible[eligible["innings"].eq(2)]
    if not second.empty:
        model_c = second["pressure_method"].astype(str).str.contains(
            "Model C", case=False, na=False
        )
        if not model_c.all():
            raise ValueError(
                "Eligible second-innings rows do not consistently use Model C."
            )
        if second["run_rate_gap"].isna().all():
            raise ValueError("Model C run_rate_gap is entirely missing.")

    # Use the maximum pre-delivery balls-remaining value as the scheduled
    # legal-ball limit for each innings. This also accommodates shortened
    # innings in the phase assignment.
    eligible["innings_max_balls"] = eligible.groupby(
        ["innings", "match_id"]
    )["legal_balls_remaining"].transform("max")
    eligible["phase"] = [
        phase_from_balls(remaining, maximum)
        for remaining, maximum in zip(
            eligible["legal_balls_remaining"],
            eligible["innings_max_balls"],
        )
    ]

    # Pooled match weights: each match contributes total weight 1
    # across all eligible rows in that innings and competition.
    eligible["pooled_row_count"] = eligible.groupby(
        ["innings", "competition_group", "match_id"]
    )["match_id"].transform("size")
    eligible["pooled_weight"] = 1.0 / eligible["pooled_row_count"]

    # Phase weights: each match contributes total weight 1 within each phase.
    eligible["phase_row_count"] = eligible.groupby(
        ["innings", "competition_group", "phase", "match_id"]
    )["match_id"].transform("size")
    eligible["phase_weight"] = 1.0 / eligible["phase_row_count"]

    pooled_check = eligible.groupby(
        ["innings", "competition_group", "match_id"]
    )["pooled_weight"].sum()
    phase_check = eligible.groupby(
        ["innings", "competition_group", "phase", "match_id"]
    )["phase_weight"].sum()
    if not np.allclose(pooled_check.to_numpy(), 1.0):
        raise ValueError("Pooled equal-match weighting validation failed.")
    if not np.allclose(phase_check.to_numpy(), 1.0):
        raise ValueError("Phase equal-match weighting validation failed.")

    print("Eligible pressure rows:", len(eligible))
    print("Eligible matches:", eligible["match_id"].nunique())
    print("Comparing pooled and phase-specific 40/75 classifications...")

    threshold_records = []
    row_records = []

    group_cols = ["innings", "competition_group"]
    for (innings, competition), group in eligible.groupby(
        group_cols, sort=True, dropna=False
    ):
        match_count = group["match_id"].nunique()
        if match_count < MIN_MATCHES:
            continue

        for role, value_col in ROLE_COLUMNS.items():
            pooled_low = weighted_quantile(
                group[value_col], LOW_PERCENTILE, group["pooled_weight"]
            )
            pooled_high = weighted_quantile(
                group[value_col], HIGH_PERCENTILE, group["pooled_weight"]
            )

            phase_thresholds = {}
            for phase, phase_group in group.groupby("phase", sort=True):
                phase_matches = phase_group["match_id"].nunique()
                if phase_matches < MIN_MATCHES:
                    continue

                phase_low = weighted_quantile(
                    phase_group[value_col],
                    LOW_PERCENTILE,
                    phase_group["phase_weight"],
                )
                phase_high = weighted_quantile(
                    phase_group[value_col],
                    HIGH_PERCENTILE,
                    phase_group["phase_weight"],
                )
                phase_thresholds[phase] = (phase_low, phase_high)

                threshold_records.append({
                    "innings": innings,
                    "competition_group": competition,
                    "role": role,
                    "phase": phase,
                    "phase_matches": phase_matches,
                    "pooled_low_threshold": pooled_low,
                    "pooled_high_threshold": pooled_high,
                    "phase_low_threshold": phase_low,
                    "phase_high_threshold": phase_high,
                    "low_threshold_difference": phase_low - pooled_low,
                    "high_threshold_difference": phase_high - pooled_high,
                })

            # Compare each delivery using pooled and phase-specific cutoffs.
            # Only rows whose phase has a supported threshold are included.
            for phase, (phase_low, phase_high) in phase_thresholds.items():
                pg = group[group["phase"].eq(phase)].copy()
                values = pg[value_col].to_numpy(dtype=float)
                pooled_class = classify(values, pooled_low, pooled_high)
                phase_class = classify(values, phase_low, phase_high)

                changed = pooled_class != phase_class
                transitions = pd.crosstab(
                    pd.Series(pooled_class, name="pooled_class"),
                    pd.Series(phase_class, name="phase_class"),
                    dropna=False,
                )

                transition_counts = {
                    f"{from_class}_to_{to_class}": int(
                        transitions.loc[from_class, to_class]
                    ) if (
                        from_class in transitions.index
                        and to_class in transitions.columns
                    ) else 0
                    for from_class in ["Low", "Moderate", "High"]
                    for to_class in ["Low", "Moderate", "High"]
                    if from_class != to_class
                }

                row_records.append({
                    "innings": innings,
                    "competition_group": competition,
                    "role": role,
                    "phase": phase,
                    "rows": len(pg),
                    "matches": pg["match_id"].nunique(),
                    "changed_rows": int(changed.sum()),
                    "unchanged_rows": int((~changed).sum()),
                    "changed_pct": safe_pct(int(changed.sum()), len(pg)),
                    "pooled_low_rows": int((pooled_class == "Low").sum()),
                    "pooled_moderate_rows": int((pooled_class == "Moderate").sum()),
                    "pooled_high_rows": int((pooled_class == "High").sum()),
                    "phase_low_rows": int((phase_class == "Low").sum()),
                    "phase_moderate_rows": int((phase_class == "Moderate").sum()),
                    "phase_high_rows": int((phase_class == "High").sum()),
                    **transition_counts,
                })

    thresholds = pd.DataFrame(threshold_records)
    row_summary = pd.DataFrame(row_records)

    # Match-level impact: how many matches have any changed delivery and
    # how much the high-pressure delivery count changes per match.
    match_records = []
    for (innings, competition) in eligible.groupby(
        group_cols, sort=True, dropna=False
    ).groups:
        group = eligible[
            eligible["innings"].eq(innings)
            & eligible["competition_group"].eq(competition)
        ]
        for role, value_col in ROLE_COLUMNS.items():
            pooled_low = weighted_quantile(
                group[value_col], LOW_PERCENTILE, group["pooled_weight"]
            )
            pooled_high = weighted_quantile(
                group[value_col], HIGH_PERCENTILE, group["pooled_weight"]
            )

            for phase, pg in group.groupby("phase", sort=True):
                phase_low = weighted_quantile(
                    pg[value_col], LOW_PERCENTILE, pg["phase_weight"]
                )
                phase_high = weighted_quantile(
                    pg[value_col], HIGH_PERCENTILE, pg["phase_weight"]
                )
                if pg["match_id"].nunique() < MIN_MATCHES:
                    continue

                tmp = pg[["match_id", "season", value_col]].copy()
                tmp["pooled_class"] = classify(
                    tmp[value_col].to_numpy(dtype=float),
                    pooled_low, pooled_high
                )
                tmp["phase_class"] = classify(
                    tmp[value_col].to_numpy(dtype=float),
                    phase_low, phase_high
                )
                tmp["changed"] = tmp["pooled_class"] != tmp["phase_class"]
                tmp["pooled_high"] = tmp["pooled_class"].eq("High")
                tmp["phase_high"] = tmp["phase_class"].eq("High")
                per_match = tmp.groupby(["match_id", "season"]).agg(
                    rows=(value_col, "size"),
                    changed_rows=("changed", "sum"),
                    pooled_high_rows=("pooled_high", "sum"),
                    phase_high_rows=("phase_high", "sum"),
                ).reset_index()
                per_match["high_row_change"] = (
                    per_match["phase_high_rows"] - per_match["pooled_high_rows"]
                )
                for _, r in per_match.iterrows():
                    match_records.append({
                        "innings": innings,
                        "competition_group": competition,
                        "role": role,
                        "phase": phase,
                        "match_id": r["match_id"],
                        "season": r["season"],
                        "rows": int(r["rows"]),
                        "changed_rows": int(r["changed_rows"]),
                        "changed_pct": safe_pct(
                            int(r["changed_rows"]), int(r["rows"])
                        ),
                        "pooled_high_rows": int(r["pooled_high_rows"]),
                        "phase_high_rows": int(r["phase_high_rows"]),
                        "high_row_change": int(r["high_row_change"]),
                    })

    match_detail = pd.DataFrame(match_records)

    if not match_detail.empty:
        match_summary = (
            match_detail.groupby(
                ["innings", "competition_group", "role", "phase"],
                dropna=False,
            )
            .agg(
                matches=("match_id", "nunique"),
                matches_with_any_change=(
                    "changed_rows", lambda s: int((s > 0).sum())
                ),
                mean_changed_pct=("changed_pct", "mean"),
                median_changed_pct=("changed_pct", "median"),
                total_pooled_high_rows=("pooled_high_rows", "sum"),
                total_phase_high_rows=("phase_high_rows", "sum"),
                total_high_row_change=("high_row_change", "sum"),
                mean_high_row_change_per_match=("high_row_change", "mean"),
            )
            .reset_index()
        )
        match_summary["matches_with_any_change_pct"] = (
            100 * match_summary["matches_with_any_change"] / match_summary["matches"]
        )
    else:
        match_summary = pd.DataFrame()

    if not match_detail.empty:
        season_summary = (
            match_detail.groupby(
                ["innings", "competition_group", "role", "phase", "season"],
                dropna=False,
            )
            .agg(
                matches=("match_id", "nunique"),
                matches_with_any_change=(
                    "changed_rows", lambda s: int((s > 0).sum())
                ),
                mean_changed_pct=("changed_pct", "mean"),
                total_pooled_high_rows=("pooled_high_rows", "sum"),
                total_phase_high_rows=("phase_high_rows", "sum"),
                total_high_row_change=("high_row_change", "sum"),
            )
            .reset_index()
        )
        season_summary["matches_with_any_change_pct"] = (
            100 * season_summary["matches_with_any_change"] / season_summary["matches"]
        )
    else:
        season_summary = pd.DataFrame()

    thresholds.to_csv(THRESHOLD_OUTPUT, index=False)
    row_summary.to_csv(ROW_OUTPUT, index=False)
    match_summary.to_csv(MATCH_OUTPUT, index=False)
    season_summary.to_csv(SEASON_OUTPUT, index=False)

    print("\\n" + "=" * 78)
    print("PRESSURE CLASSIFICATION SENSITIVITY CHECK COMPLETE")
    print("=" * 78)
    print("\\nPooled vs phase-specific thresholds:")
    if not thresholds.empty:
        print(thresholds.to_string(index=False))
    print("\\nDelivery-level classification changes:")
    if not row_summary.empty:
        print(row_summary.to_string(index=False))
    print("\\nMatch-level impact:")
    if not match_summary.empty:
        print(match_summary.to_string(index=False))
    print("\\nSeason-level impact:")
    if not season_summary.empty:
        print(season_summary.to_string(index=False))

    print("\\nFiles saved:")
    for path in [
        THRESHOLD_OUTPUT, ROW_OUTPUT, MATCH_OUTPUT, SEASON_OUTPUT
    ]:
        print(path)


if __name__ == "__main__":
    main()
