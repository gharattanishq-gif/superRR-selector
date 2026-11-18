from pathlib import Path
import numpy as np
import pandas as pd

# SupeRR Selector Season 2 — Round 1
# Reconstruct pressure-specific bowling dot-ball metrics from source deliveries.
# Uses the locked 40/75 calibration policy and validates against the existing
# player-match pressure bowling aggregation before writing outputs.


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
CAL = ROOT / "outputs" / "analysis" / "pressure_calibration"
OUT = ROOT / "outputs" / "analysis" / "pressure_index_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

PATHS = {
    "balls": DATA / "ball_by_ball.csv",
    "matches": DATA / "matches.csv",
    "states": PROC / "pressure_states.csv",
    "phase_thresholds": CAL / "pressure_phase_threshold_comparison.csv",
    "pooled_thresholds": CAL / "pressure_threshold_comparison.csv",
    "reference_bowling": PROC / "player_match_pressure_bowling.csv",
}
for label, path in PATHS.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} input: {path}")

KEY = ["match_id", "innings", "overid", "over_ball_num"]
KNOWN_EXTRA_TYPES = {"", "wide", "overstep", "byes", "legbyes"}

# Locked policy: use phase-specific thresholds only in the listed strata;
# otherwise use the pooled innings x competition threshold.
THRESHOLD_POLICY = {
    (1, "IPL", "Powerplay"): "phase",
    (1, "IPL", "Middle"): "pooled",
    (1, "IPL", "Death"): "phase",
    (1, "T20I", "Powerplay"): "phase",
    (1, "T20I", "Middle"): "pooled",
    (1, "T20I", "Death"): "pooled",
    (2, "IPL", "Powerplay"): "phase",
    (2, "IPL", "Middle"): "pooled",
    (2, "IPL", "Death"): "phase",
    (2, "T20I", "Powerplay"): "phase",
    (2, "T20I", "Middle"): "pooled",
    (2, "T20I", "Death"): "phase",
}


def require(frame, columns, label):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def innings_phase(innings_max_balls, balls_remaining):
    if pd.isna(innings_max_balls) or pd.isna(balls_remaining) or innings_max_balls <= 0:
        return "Unknown"
    balls_bowled = innings_max_balls - balls_remaining
    powerplay_end = int(np.ceil(innings_max_balls * 0.25))
    middle_end = int(np.ceil(innings_max_balls * 0.75))
    if balls_bowled < powerplay_end:
        return "Powerplay"
    if balls_bowled < middle_end:
        return "Middle"
    return "Death"


print("Loading source files...")
balls = pd.read_csv(PATHS["balls"], low_memory=False)
states = pd.read_csv(PATHS["states"], low_memory=False)
matches = pd.read_csv(PATHS["matches"], low_memory=False)
phase_thresholds = pd.read_csv(PATHS["phase_thresholds"], low_memory=False)
pooled_thresholds = pd.read_csv(PATHS["pooled_thresholds"], low_memory=False)
reference = pd.read_csv(PATHS["reference_bowling"], low_memory=False)

require(balls, KEY + ["bowler_id", "bowler_name", "runs", "extras_type", "extras1_runs", "extras_2_runs"], "ball_by_ball")
require(states, KEY + ["match_id", "match_date", "competition_group", "season", "innings", "legal_balls_remaining", "batting_pressure", "bowling_pressure", "pressure_eligible", "bowler_id", "bowler_name"], "pressure_states")
require(matches, ["match_id", "first_inn_revised_overs", "second_inn_revised_overs"], "matches")
require(phase_thresholds, ["innings", "competition_group", "phase", "role", "scheme", "phase_low_threshold", "phase_high_threshold"], "phase thresholds")
require(pooled_thresholds, ["innings", "competition_group", "role", "scheme", "low_threshold", "high_threshold"], "pooled thresholds")
require(reference, ["match_id", "innings", "bowler_id", "phase", "pressure_level", "pressure_deliveries", "valid_bowling_deliveries", "legal_balls", "ambiguous_deliveries"], "reference pressure bowling")

for frame in [balls, states, reference]:
    frame["innings"] = pd.to_numeric(frame["innings"], errors="coerce")

# Delivery key integrity.
for label, frame in [("ball_by_ball", balls), ("pressure_states", states)]:
    duplicates = int(frame.duplicated(KEY).sum())
    if duplicates:
        raise ValueError(f"{label} has {duplicates:,} duplicate delivery keys.")
    print(f"{label} rows: {len(frame):,}; duplicate keys: {duplicates:,}")

# Validate scheduled legal-ball limits. The revised-overs fields in this dataset
# already store legal-ball limits, so do not multiply them by six.
limit_checks = []
for innings, source_col in [(1, "first_inn_revised_overs"), (2, "second_inn_revised_overs")]:
    state_max = (
        states.loc[states["innings"].eq(innings)]
        .groupby("match_id")["legal_balls_remaining"].max()
        .rename("states_max_balls")
    )
    match_limits = matches[["match_id", source_col]].copy()
    match_limits[source_col] = pd.to_numeric(match_limits[source_col], errors="coerce").fillna(120)
    check = match_limits.merge(state_max, on="match_id", how="inner")
    check["matches_limit"] = check[source_col].eq(check["states_max_balls"])
    limit_checks.append({
        "innings": innings,
        "matches_checked": len(check),
        "exact_limit_matches": int(check["matches_limit"].sum()),
        "mismatched_limit_matches": int((~check["matches_limit"]).sum()),
    })
    if not check["matches_limit"].all():
        raise ValueError(f"Innings {innings}: scheduled legal-ball limit mismatch detected.")
limit_validation = pd.DataFrame(limit_checks)

# Reconstruct 25% / 50% / 25% phases using each match-innings' scheduled limit.
states["legal_balls_remaining"] = pd.to_numeric(states["legal_balls_remaining"], errors="coerce")
states["innings_max_balls"] = states.groupby(["match_id", "innings"])["legal_balls_remaining"].transform("max")
states["phase"] = [
    innings_phase(mx, rem)
    for mx, rem in zip(states["innings_max_balls"], states["legal_balls_remaining"])
]

# Load only the locked 40/75 thresholds.
phase_thresholds = phase_thresholds[phase_thresholds["scheme"].eq("40_75")].copy()
pooled_thresholds = pooled_thresholds[pooled_thresholds["scheme"].eq("40_75")].copy()
phase_lookup = {}
for row in phase_thresholds.itertuples(index=False):
    phase_lookup[(int(row.innings), row.competition_group, row.phase, row.role)] = (
        float(row.phase_low_threshold), float(row.phase_high_threshold)
    )
pooled_lookup = {}
for row in pooled_thresholds.itertuples(index=False):
    pooled_lookup[(int(row.innings), row.competition_group, row.role)] = (
        float(row.low_threshold), float(row.high_threshold)
    )

for role, value_col in [("Batting", "batting_pressure"), ("Bowling", "bowling_pressure")]:
    lows, highs, threshold_types = [], [], []
    for row in states[["innings", "competition_group", "phase"]].itertuples(index=False):
        key = (int(row.innings), row.competition_group, row.phase)
        threshold_type = THRESHOLD_POLICY.get(key)
        if threshold_type is None:
            lows.append(np.nan); highs.append(np.nan); threshold_types.append("unsupported")
            continue
        if threshold_type == "phase":
            threshold_key = (int(row.innings), row.competition_group, row.phase, role)
            threshold = phase_lookup.get(threshold_key)
        else:
            threshold_key = (int(row.innings), row.competition_group, role)
            threshold = pooled_lookup.get(threshold_key)
        if threshold is None:
            raise ValueError(f"Missing locked threshold for {threshold_key}")
        lows.append(threshold[0]); highs.append(threshold[1]); threshold_types.append(threshold_type)
    prefix = role.lower()
    states[f"{prefix}_low_threshold"] = lows
    states[f"{prefix}_high_threshold"] = highs
    states[f"{prefix}_threshold_type"] = threshold_types
    states[f"{prefix}_pressure_level"] = np.select(
        [states[value_col] < states[f"{prefix}_low_threshold"],
         states[value_col] > states[f"{prefix}_high_threshold"]],
        ["Low", "High"], default="Moderate"
    )

if states[["batting_pressure_level", "bowling_pressure_level"]].isna().any().any():
    raise ValueError("Pressure classification produced missing levels.")

# Exact delivery-level join.
state_columns = KEY + [
    "match_date", "comp_name", "season", "competition_group", "pressure_eligible",
    "chase_prediction_supported", "phase", "batting_pressure_level", "bowling_pressure_level",
]
joined = balls.merge(
    states[state_columns], on=KEY, how="inner", validate="one_to_one", suffixes=("_ball", "_state")
)
if len(joined) != len(states):
    raise ValueError(
        f"Delivery join incomplete: joined {len(joined):,} of {len(states):,} pressure-state rows."
    )
matched_pressure_deliveries = len(joined)

# Match the existing metric-builder's bowling ambiguity and legal-ball rules.
for col in ["runs", "extras1_runs", "extras_2_runs"]:
    joined[col] = pd.to_numeric(joined[col], errors="coerce").fillna(0)
joined["extras_clean"] = joined["extras_type"].fillna("").astype(str).str.strip().str.lower()
missing_type_with_extras = joined["extras_clean"].eq("") & (
    joined["extras1_runs"].ne(0) | joined["extras_2_runs"].ne(0)
)
ambiguous_type = joined["extras_clean"].isin(["other", "penalty"])
unknown_type = ~joined["extras_clean"].isin(KNOWN_EXTRA_TYPES)
joined["bowling_unresolved"] = (
    missing_type_with_extras | ambiguous_type | unknown_type | joined["extras_2_runs"].ne(0)
)
joined["is_legal"] = ~joined["extras_clean"].isin(["wide", "overstep"])
joined["valid_bowling_delivery"] = ~joined["bowling_unresolved"]
joined["valid_legal_ball"] = joined["is_legal"] & joined["valid_bowling_delivery"]
joined["team_runs_delivery"] = joined["runs"] + joined["extras1_runs"] + joined["extras_2_runs"]
joined["dot_ball"] = joined["valid_legal_ball"] & joined["team_runs_delivery"].eq(0)

# Only pressure-eligible states enter the pressure index.
joined = joined[joined["pressure_eligible"].fillna(False)].copy()
joined["legal_ball_count"] = joined["valid_legal_ball"].astype(int)
joined["dot_ball_count"] = joined["dot_ball"].astype(int)
joined["ambiguous_delivery_count"] = joined["bowling_unresolved"].astype(int)
joined["valid_bowling_delivery_count"] = joined["valid_bowling_delivery"].astype(int)

# Pressure-level bowling aggregation. One row per player-match-innings-phase-level.
group_cols = [
    "match_id", "match_date", "comp_name", "season", "competition_group", "innings",
    "phase", "bowler_id", "bowler_name", "bowling_pressure_level",
]
result = joined.groupby(group_cols, dropna=False).agg(
    pressure_deliveries=("match_id", "size"),
    valid_bowling_deliveries=("valid_bowling_delivery_count", "sum"),
    legal_balls=("legal_ball_count", "sum"),
    dot_balls=("dot_ball_count", "sum"),
    ambiguous_deliveries=("ambiguous_delivery_count", "sum"),
).reset_index().rename(columns={"bowling_pressure_level": "pressure_level"})
result["dot_ball_pct"] = 100 * result["dot_balls"] / result["legal_balls"].replace(0, np.nan)
result["dot_ball_pct_complete"] = result["legal_balls"].gt(0) & result["ambiguous_deliveries"].eq(0)

# Reconcile all aggregate counts with the existing pressure-bowling output.
ref = reference.copy()
ref["match_id"] = pd.to_numeric(ref["match_id"], errors="coerce")
result["match_id"] = pd.to_numeric(result["match_id"], errors="coerce")
recon_keys = ["match_id", "innings", "bowler_id", "phase", "pressure_level"]
comparison = result.merge(
    ref[recon_keys + ["pressure_deliveries", "valid_bowling_deliveries", "legal_balls", "ambiguous_deliveries"]],
    on=recon_keys, how="outer", suffixes=("_reconstructed", "_reference"), indicator=True,
    validate="one_to_one"
)
if not comparison["_merge"].eq("both").all():
    raise ValueError("Pressure-bowling aggregate keys do not reconcile with the reference file.")
count_cols = ["pressure_deliveries", "valid_bowling_deliveries", "legal_balls", "ambiguous_deliveries"]
for col in count_cols:
    left = comparison[f"{col}_reconstructed"]
    right = comparison[f"{col}_reference"]
    if not left.equals(right):
        # Numeric equality with null-safe comparison.
        if not np.array_equal(left.fillna(-999999).to_numpy(), right.fillna(-999999).to_numpy()):
            raise ValueError(f"Reconciliation failed for {col}.")

validation = pd.DataFrame([
    {"check": "pressure_state_rows", "value": len(states)},
    {"check": "matched_pressure_deliveries", "value": matched_pressure_deliveries},
    {"check": "unmatched_pressure_deliveries", "value": len(states) - matched_pressure_deliveries},
    {"check": "pressure_eligible_deliveries", "value": int(joined["pressure_eligible"].fillna(False).sum())},
    {"check": "ambiguous_bowling_deliveries", "value": int(joined["bowling_unresolved"].sum())},
    {"check": "valid_legal_balls", "value": int(joined["valid_legal_ball"].sum())},
    {"check": "dot_balls", "value": int(joined["dot_ball"].sum())},
    {"check": "player_match_phase_pressure_rows", "value": len(result)},
    {"check": "zero_legal_ball_rows", "value": int(result["legal_balls"].eq(0).sum())},
    {"check": "aggregate_key_mismatches", "value": int((comparison["_merge"] != "both").sum())},
])

result_path = OUT / "player_match_pressure_bowling_dotballs.csv"
validation_path = OUT / "pressure_bowling_dotball_validation.csv"
limit_path = OUT / "pressure_phase_limit_validation.csv"
result.to_csv(result_path, index=False)
validation.to_csv(validation_path, index=False)
limit_validation.to_csv(limit_path, index=False)

print("\nPRESSURE BOWLING DOT-BALL RECONSTRUCTION COMPLETE")
print("Output:", result_path)
print("Validation:", validation_path)
print("Phase limits:")
print(limit_validation.to_string(index=False))
print("\nValidation summary:")
print(validation.to_string(index=False))
print("\nDot-ball percentage by competition / innings / pressure level:")
summary = result.groupby(["competition_group", "innings", "pressure_level"], dropna=False).agg(
    legal_balls=("legal_balls", "sum"), dot_balls=("dot_balls", "sum")
).reset_index()
summary["dot_ball_pct"] = 100 * summary["dot_balls"] / summary["legal_balls"].replace(0, np.nan)
print(summary.to_string(index=False))
