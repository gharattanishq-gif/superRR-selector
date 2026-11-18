from pathlib import Path
import numpy as np
import pandas as pd

# SupeRR Selector Season 2 — Round 1
# Integrated pressure-component diagnostics.
# This script builds comparable component diagnostics only; it does NOT lock
# pressure-index weights, apply reliability shrinkage, or create final rankings.

def find_root():
    here = Path(__file__).resolve()
    for candidate in [Path.cwd(), *here.parents]:
        if (candidate / "outputs" / "analysis" / "player_metrics").exists():
            return candidate
    raise FileNotFoundError(
        "Project root not found. Put this script in SupeRR/src and run it from the project root."
    )

ROOT = find_root()
METRICS = ROOT / "outputs" / "analysis" / "player_metrics"
PRESSURE = ROOT / "outputs" / "analysis" / "pressure_index_diagnostics"
OUT = ROOT / "outputs" / "analysis" / "integrated_pressure_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "bat_player": PRESSURE / "batting_pressure_player_diagnostics.csv",
    "bat_strata": PRESSURE / "batting_pressure_innings_phase_diagnostics.csv",
    "bat_matches": PRESSURE / "batting_high_pressure_match_diagnostics.csv",
    "bowl_player": PRESSURE / "bowling_pressure_player_diagnostics.csv",
    "bowl_strata": PRESSURE / "bowling_pressure_innings_phase_diagnostics.csv",
    "bowl_matches": PRESSURE / "bowling_high_pressure_match_diagnostics.csv",
    "bowl_dots": PRESSURE / "player_match_pressure_bowling_dotballs.csv",
    "bat_metrics": METRICS / "player_batting_metrics.csv",
    "bowl_metrics": METRICS / "player_bowling_metrics.csv",
}

for label, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")

def read(label):
    return pd.read_csv(FILES[label], low_memory=False)

def bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.fillna("").astype(str).str.strip().str.lower().isin(
        ["true", "1", "yes", "y", "t"]
    )

def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a.div(b.where(b.ne(0)))

def percentile(s, higher_is_better=True):
    x = pd.to_numeric(s, errors="coerce")
    if not higher_is_better:
        x = -x
    return x.rank(method="average", pct=True) * 100

def mad(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 2:
        return np.nan
    med = x.median()
    return float((x - med).abs().median())

def write_csv(df, name):
    df.to_csv(OUT / name, index=False)

# Common identifiers
keys_bat = ["competition_group", "evaluation_period", "striker_id"]
keys_bowl = ["competition_group", "evaluation_period", "bowler_id"]
strata_bat = keys_bat + ["innings", "phase"]
strata_bowl = keys_bowl + ["innings", "phase"]

# ---------------------------------------------------------------------------
# BATTER COMPONENTS
# ---------------------------------------------------------------------------
bp = read("bat_player")
bm = read("bat_metrics")
bstr = read("bat_strata")
bmatch = read("bat_matches")

# Player-level eligible population is the authoritative denominator.
bm["competition_group"] = bm["population"].replace({"Senior men's T20I": "T20I"})
bm["evaluation_period"] = bm["evaluation_period"].astype(str)
bm["eligible_by_matches"] = bool_series(bm["eligible_by_matches"])
eligible_bat = bm.loc[bm["eligible_by_matches"]].copy()
eligible_bat = eligible_bat.rename(columns={
    "striker_id": "player_id", "striker_name": "player_name"
})

bp["evaluation_period"] = bp["evaluation_period"].astype(str)
bp = bp.rename(columns={"striker_id": "player_id", "striker_name": "player_name"})
bmatch["evaluation_period"] = bmatch["evaluation_period"].astype(str)

bat_cols = ["competition_group", "evaluation_period", "player_id"] + [
    "high_pressure_deliveries", "high_balls_faced", "high_runs",
    "high_dismissals", "high_complete_balls", "high_complete_runs",
    "high_complete_dismissals", "high_pressure_strike_rate",
    "high_pressure_average", "baseline_strike_rate", "baseline_average",
    "pressure_sr_uplift", "high_pressure_matches",
    "high_pressure_match_count_for_consistency", "match_sr_mean",
    "match_sr_sd", "match_sr_median", "match_sr_mad",
    "high_pressure_sr_coverage_pct"
]
bat_cols = [c for c in bat_cols if c in bp.columns]
bat = eligible_bat.merge(
    bp[bat_cols],
    on=["competition_group", "evaluation_period", "player_id"],
    how="left", validate="one_to_one"
)
bat = bat.rename(columns={"player_id": "striker_id", "player_name": "striker_name"})

# Use complete-ball SR uplift and complete high-pressure average as supplied
# by the validated pressure diagnostics.
bat["pressure_average_uplift"] = (
    bat["high_pressure_average"] - bat["baseline_average"]
)
bat["pressure_sr_uplift"] = pd.to_numeric(
    bat.get("pressure_sr_uplift"), errors="coerce"
)
bat["high_pressure_strike_rate"] = pd.to_numeric(
    bat.get("high_pressure_strike_rate"), errors="coerce"
)
bat["high_pressure_average"] = pd.to_numeric(
    bat.get("high_pressure_average"), errors="coerce"
)

# High-pressure boundary-run percentage by innings/phase, weighted by
# high-pressure runs within the comparable strata.
bstr["evaluation_period"] = bstr["evaluation_period"].astype(str)
bstr["high_boundary_run_pct"] = pd.to_numeric(
    bstr["high_boundary_run_pct"], errors="coerce"
)
bstr["high_runs"] = pd.to_numeric(bstr["high_runs"], errors="coerce")
bstr["high_balls_faced"] = pd.to_numeric(bstr["high_balls_faced"], errors="coerce")
bstr["high_boundary_weight"] = bstr["high_runs"].clip(lower=0)
bstr["boundary_weighted"] = bstr["high_boundary_run_pct"] * bstr["high_boundary_weight"]
b_boundary = bstr.groupby(keys_bat, dropna=False).agg(
    high_boundary_weight=("high_boundary_weight", "sum"),
    boundary_weighted=("boundary_weighted", "sum"),
).reset_index()
b_boundary["high_pressure_boundary_run_pct"] = safe_div(
    b_boundary["boundary_weighted"], b_boundary["high_boundary_weight"]
)
bat = bat.merge(
    b_boundary[keys_bat + ["high_pressure_boundary_run_pct"]],
    on=keys_bat, how="left", validate="one_to_one"
)

# Match consistency: retain raw support and robust spread. The source file is
# already one row per player-match; recompute counts to validate it.
bmatch["match_high_strike_rate"] = pd.to_numeric(
    bmatch["match_high_strike_rate"], errors="coerce"
)
bcons = bmatch.groupby(keys_bat, dropna=False).agg(
    match_rows=("match_id", "nunique"),
    match_sr_sd_recalc=("match_high_strike_rate", "std"),
    match_sr_mad_recalc=("match_high_strike_rate", mad),
).reset_index()
bat = bat.merge(bcons, on=keys_bat, how="left", validate="one_to_one")
bat["batting_match_count_check"] = (
    bat["high_pressure_match_count_for_consistency"].fillna(0)
    == bat["match_rows"].fillna(0)
)

# Explicit support indicators. No arbitrary cutoff is imposed here.
bat["high_pressure_balls_support"] = pd.to_numeric(
    bat["high_balls_faced"], errors="coerce"
)
bat["pressure_exposure_band"] = np.select(
    [
        bat["high_pressure_matches"].fillna(0).lt(1),
        bat["high_pressure_matches"].fillna(0).between(1, 2),
        bat["high_pressure_matches"].fillna(0).between(3, 5),
        bat["high_pressure_matches"].fillna(0).ge(6),
    ],
    ["No high-pressure matches", "1-2 matches", "3-5 matches", "6+ matches"],
    default="Unknown"
)

# ---------------------------------------------------------------------------
# BOWLER COMPONENTS
# ---------------------------------------------------------------------------
bpw = read("bowl_player")
bowlm = read("bowl_metrics")
bowlstr = read("bowl_strata")
bowlmatch = read("bowl_matches")
dots = read("bowl_dots")

bowlm["competition_group"] = bowlm["population"].replace({"Senior men's T20I": "T20I"})
bowlm["evaluation_period"] = bowlm["evaluation_period"].astype(str)
bowlm["eligible_by_matches"] = bool_series(bowlm["eligible_by_matches"])
eligible_bowl = bowlm.loc[bowlm["eligible_by_matches"]].copy()
eligible_bowl = eligible_bowl.rename(columns={
    "bowler_id": "player_id", "bowler_name": "player_name"
})

bpw = bpw.rename(columns={"bowler_id": "player_id", "bowler_name": "player_name"})
bpw["evaluation_period"] = bpw["evaluation_period"].astype(str)
bowlmatch["evaluation_period"] = bowlmatch["evaluation_period"].astype(str)
bowlstr["evaluation_period"] = bowlstr["evaluation_period"].astype(str)
dots["season"] = dots["season"].astype(str)
dots["evaluation_period"] = np.where(
    dots["competition_group"].eq("IPL"), dots["season"], "2022-2026 combined"
).astype(str)

bowl_cols = ["competition_group", "evaluation_period", "player_id"] + [
    "high_pressure_deliveries", "high_valid_deliveries", "high_legal_balls",
    "high_runs_conceded", "high_wickets", "high_ambiguous_deliveries",
    "baseline_legal_balls", "baseline_runs_conceded", "baseline_wickets",
    "high_pressure_economy", "high_pressure_bowling_average",
    "high_pressure_wickets_per_6_legal_balls", "baseline_economy",
    "baseline_wickets_per_6_legal_balls", "pressure_economy_uplift",
    "pressure_wicket_rate_uplift", "high_pressure_match_count_for_consistency",
    "match_economy_mean", "match_economy_sd", "match_economy_median",
    "match_economy_mad", "high_pressure_matches",
    "high_pressure_delivery_coverage_pct"
]
bowl_cols = [c for c in bowl_cols if c in bpw.columns]
bowl = eligible_bowl.merge(
    bpw[bowl_cols],
    on=["competition_group", "evaluation_period", "player_id"],
    how="left", validate="one_to_one"
)
bowl = bowl.rename(columns={"player_id": "bowler_id", "player_name": "bowler_name"})

# Build High vs Low+Moderate dot-ball uplift within matching innings/phase.
dots["innings"] = pd.to_numeric(dots["innings"], errors="coerce")
dots["legal_balls"] = pd.to_numeric(dots["legal_balls"], errors="coerce")
dots["dot_balls"] = pd.to_numeric(dots["dot_balls"], errors="coerce")
dots["dot_ball_pct"] = pd.to_numeric(dots["dot_ball_pct"], errors="coerce")
dots["bowler_id"] = pd.to_numeric(dots["bowler_id"], errors="coerce")
dots["evaluation_period"] = dots["evaluation_period"].astype(str)
dot_keys = ["competition_group", "evaluation_period", "bowler_id", "innings", "phase"]
dot_pivot = dots.pivot_table(
    index=dot_keys, columns="pressure_level",
    values=["legal_balls", "dot_balls"], aggfunc="sum"
)
dot_pivot.columns = ["_".join([str(x) for x in c if str(x)]) for c in dot_pivot.columns]
dot_pivot = dot_pivot.reset_index()
for c in ["legal_balls_High", "dot_balls_High", "legal_balls_Low",
          "dot_balls_Low", "legal_balls_Moderate", "dot_balls_Moderate"]:
    if c not in dot_pivot:
        dot_pivot[c] = np.nan

dot_pivot["baseline_legal_balls"] = (
    dot_pivot["legal_balls_Low"].fillna(0)
    + dot_pivot["legal_balls_Moderate"].fillna(0)
)
dot_pivot["baseline_dot_balls"] = (
    dot_pivot["dot_balls_Low"].fillna(0)
    + dot_pivot["dot_balls_Moderate"].fillna(0)
)
dot_pivot["high_dot_pct"] = safe_div(
    dot_pivot["dot_balls_High"], dot_pivot["legal_balls_High"]
) * 100
dot_pivot["baseline_dot_pct"] = safe_div(
    dot_pivot["baseline_dot_balls"], dot_pivot["baseline_legal_balls"]
) * 100
dot_pivot["dot_ball_uplift"] = (
    dot_pivot["high_dot_pct"] - dot_pivot["baseline_dot_pct"]
)
dot_pivot["stratum_has_comparison"] = (
    dot_pivot["legal_balls_High"].gt(0)
    & dot_pivot["baseline_legal_balls"].gt(0)
    & dot_pivot["dot_ball_uplift"].notna()
)

# Weight comparable stratum uplifts by High-pressure legal balls, not baseline
# exposure. This avoids allowing a huge baseline sample to dominate the uplift.
dot_pivot["uplift_weight"] = np.where(
    dot_pivot["stratum_has_comparison"],
    dot_pivot["legal_balls_High"], 0
)
dot_pivot["weighted_uplift"] = (
    dot_pivot["dot_ball_uplift"] * dot_pivot["uplift_weight"]
)
dot_player = dot_pivot.groupby(
    ["competition_group", "evaluation_period", "bowler_id"], dropna=False
).agg(
    high_pressure_dot_legal_balls=("legal_balls_High", "sum"),
    high_pressure_dot_balls=("dot_balls_High", "sum"),
    baseline_dot_legal_balls=("baseline_legal_balls", "sum"),
    baseline_dot_balls=("baseline_dot_balls", "sum"),
    dot_uplift_comparable_strata=("stratum_has_comparison", "sum"),
    dot_uplift_weight=("uplift_weight", "sum"),
    dot_uplift_weighted_sum=("weighted_uplift", "sum"),
).reset_index()
dot_player["high_pressure_dot_ball_pct"] = safe_div(
    dot_player["high_pressure_dot_balls"],
    dot_player["high_pressure_dot_legal_balls"]
) * 100
dot_player["baseline_dot_ball_pct"] = safe_div(
    dot_player["baseline_dot_balls"],
    dot_player["baseline_dot_legal_balls"]
) * 100
dot_player["pressure_dot_ball_uplift"] = safe_div(
    dot_player["dot_uplift_weighted_sum"], dot_player["dot_uplift_weight"]
)
bowl = bowl.merge(
    dot_player,
    on=keys_bowl,
    how="left",
    validate="one_to_one"
)

# Recompute match-level consistency and check against player diagnostic.
bowlmatch["match_high_economy"] = pd.to_numeric(
    bowlmatch["match_high_economy"], errors="coerce"
)
bcons = bowlmatch.groupby(keys_bowl, dropna=False).agg(
    match_rows=("match_id", "nunique"),
    match_economy_sd_recalc=("match_high_economy", "std"),
    match_economy_mad_recalc=("match_high_economy", mad),
).reset_index()
bowl = bowl.merge(bcons, on=keys_bowl, how="left", validate="one_to_one")
bowl["bowling_match_count_check"] = (
    bowl["high_pressure_match_count_for_consistency"].fillna(0)
    == bowl["match_rows"].fillna(0)
)
bowl["pressure_exposure_band"] = np.select(
    [
        bowl["high_pressure_matches"].fillna(0).lt(1),
        bowl["high_pressure_matches"].fillna(0).between(1, 2),
        bowl["high_pressure_matches"].fillna(0).between(3, 5),
        bowl["high_pressure_matches"].fillna(0).ge(6),
    ],
    ["No high-pressure matches", "1-2 matches", "3-5 matches", "6+ matches"],
    default="Unknown"
)

# ---------------------------------------------------------------------------
# DIAGNOSTIC SCORES: percentile components within each population/period.
# These are descriptive component scales, NOT combined PPI scores.
# ---------------------------------------------------------------------------
for df, specs, group_cols in [
    (bat, [
        ("high_pressure_strike_rate", True, "abs_high_sr_pctile"),
        ("high_pressure_average", True, "abs_high_average_pctile"),
        ("high_pressure_boundary_run_pct", True, "abs_high_boundary_pctile"),
        ("pressure_sr_uplift", True, "sr_uplift_pctile"),
        ("pressure_average_uplift", True, "average_uplift_pctile"),
        ("match_sr_mad_recalc", False, "sr_consistency_pctile"),
    ], ["competition_group", "evaluation_period"]),
    (bowl, [
        ("high_pressure_economy", False, "abs_high_economy_pctile"),
        ("high_pressure_bowling_average", False, "abs_high_average_pctile"),
        ("high_pressure_wickets_per_6_legal_balls", True, "abs_high_wicket_rate_pctile"),
        ("high_pressure_dot_ball_pct", True, "abs_high_dot_pctile"),
        ("pressure_economy_uplift", True, "economy_uplift_pctile"),
        ("pressure_wicket_rate_uplift", True, "wicket_uplift_pctile"),
        ("pressure_dot_ball_uplift", True, "dot_uplift_pctile"),
        ("match_economy_mad_recalc", False, "economy_consistency_pctile"),
    ], ["competition_group", "evaluation_period"]),
]:
    for source, higher, target in specs:
        df[target] = df.groupby(group_cols, dropna=False)[source].transform(
            lambda x: percentile(x, higher)
        )

# Coverage summaries; eligible player units remain denominator.
def coverage_rows(df, role, metric_cols, support_cols):
    rows = []
    for (comp, period), g in df.groupby(
        ["competition_group", "evaluation_period"], dropna=False
    ):
        row = {
            "role": role, "competition_group": comp,
            "evaluation_period": period, "eligible_player_units": len(g)
        }
        for col in metric_cols:
            row[f"{col}_available"] = int(pd.to_numeric(g[col], errors="coerce").notna().sum())
            row[f"{col}_coverage_pct"] = (
                100 * row[f"{col}_available"] / len(g) if len(g) else np.nan
            )
        for col in support_cols:
            row[f"{col}_median"] = pd.to_numeric(g[col], errors="coerce").median()
        rows.append(row)
    return rows

coverage = coverage_rows(
    bat, "Batting",
    ["high_pressure_strike_rate", "high_pressure_average",
     "high_pressure_boundary_run_pct", "pressure_sr_uplift",
     "pressure_average_uplift", "match_sr_mad_recalc"],
    ["high_pressure_matches", "high_pressure_balls_support"]
) + coverage_rows(
    bowl, "Bowling",
    ["high_pressure_economy", "high_pressure_bowling_average",
     "high_pressure_wickets_per_6_legal_balls", "high_pressure_dot_ball_pct",
     "pressure_economy_uplift", "pressure_wicket_rate_uplift",
     "pressure_dot_ball_uplift", "match_economy_mad_recalc"],
    ["high_pressure_matches", "high_legal_balls"]
)
coverage = pd.DataFrame(coverage)

# Validation
validation = [
    ("eligible_batting_player_units", len(bat)),
    ("eligible_bowling_player_units", len(bowl)),
    ("batting_duplicate_player_units", int(bat.duplicated(keys_bat).sum())),
    ("bowling_duplicate_player_units", int(bowl.duplicated(keys_bowl).sum())),
    ("batting_match_count_check_failures", int((~bat["batting_match_count_check"].fillna(False)).sum())),
    ("bowling_match_count_check_failures", int((~bowl["bowling_match_count_check"].fillna(False)).sum())),
    ("bowling_dot_player_units_with_high_pressure_balls", int(bowl["high_pressure_dot_legal_balls"].fillna(0).gt(0).sum())),
    ("bowling_dot_player_units_with_comparable_uplift", int(bowl["pressure_dot_ball_uplift"].notna().sum())),
    ("bowling_dot_uplift_strata_total", int(dot_pivot["stratum_has_comparison"].sum())),
    ("bowling_dot_rows_without_comparison", int((~dot_pivot["stratum_has_comparison"]).sum())),
]
validation = pd.DataFrame(validation, columns=["check", "value"])

write_csv(bat, "integrated_batting_pressure_components.csv")
write_csv(bowl, "integrated_bowling_pressure_components.csv")
write_csv(dot_pivot, "bowling_dotball_innings_phase_uplift.csv")
write_csv(coverage, "pressure_component_coverage_summary.csv")
write_csv(validation, "integrated_pressure_validation.csv")

print("\nINTEGRATED PRESSURE COMPONENT DIAGNOSTICS COMPLETE")
print(f"Output folder: {OUT}")
print("\nCOVERAGE SUMMARY")
print(coverage.to_string(index=False))
print("\nVALIDATION")
print(validation.to_string(index=False))
print("\nNo composite PPI scores, final weights, reliability shrinkage, or final rankings were created.")
