from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# FINAL CHASE MODEL CHECK: ORIGINAL SUPPORT RULES
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data Pack"
OUTPUT_DIR = ROOT / "outputs" / "analysis" / "final_supported_chase_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BALLS_FILE = DATA_DIR / "ball_by_ball.csv"
MATCHES_FILE = DATA_DIR / "matches.csv"
COMPETITIONS_FILE = DATA_DIR / "competitions.csv"

FIT_END = pd.Timestamp("2025-04-07")
CAL_START = pd.Timestamp("2025-04-08")
CAL_END = pd.Timestamp("2025-10-31")
TEST_START = pd.Timestamp("2025-11-01")

EXPECTED_MATCHES = {"fit": 386, "calibration": 97, "test": 121}
CHECKPOINT_INTERVAL = 6
MIN_SUPPORT_MATCHES = 5
BOOTSTRAP_ITERATIONS = 1000
RANDOM_SEED = 42

BASE_NUMERIC = [
    "wickets_lost",
    "legal_balls_remaining",
    "required_run_rate",
    "wickets_x_phase_1_12",
    "wickets_x_phase_13_30",
    "wickets_x_phase_61_90",
    "wickets_x_phase_91_120",
    "wickets_x_t20i",
]
CATEGORICAL = ["comp_name", "balls_band", "competition_group"]

MODELS = {
    "A_baseline": BASE_NUMERIC,
    "C_shared_gap": BASE_NUMERIC + ["run_rate_gap"],
}


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def competition_group(name):
    return "IPL" if "ipl" in str(name).lower() else "T20I"


def balls_band(value):
    if value <= 12:
        return "1-12"
    if value <= 30:
        return "13-30"
    if value <= 60:
        return "31-60"
    if value <= 90:
        return "61-90"
    return "91-120"


def add_interactions(data):
    data = data.copy()
    data["wickets_x_phase_1_12"] = (
        data["wickets_lost"] * (data["balls_band"] == "1-12").astype(int)
    )
    data["wickets_x_phase_13_30"] = (
        data["wickets_lost"] * (data["balls_band"] == "13-30").astype(int)
    )
    data["wickets_x_phase_61_90"] = (
        data["wickets_lost"] * (data["balls_band"] == "61-90").astype(int)
    )
    data["wickets_x_phase_91_120"] = (
        data["wickets_lost"] * (data["balls_band"] == "91-120").astype(int)
    )
    data["wickets_x_t20i"] = (
        data["wickets_lost"] * (data["competition_group"] == "T20I").astype(int)
    )
    return data


def equal_match_weights(data):
    counts = data.groupby("match_id")["match_id"].transform("size")
    return 1.0 / counts


def build_model(numeric_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_SEED)),
    ])


def metric_values(frame, probability):
    y = frame["chase_success"].to_numpy(dtype=int)
    p = np.clip(frame[probability].to_numpy(dtype=float), 1e-8, 1 - 1e-8)
    w = frame["match_weight"].to_numpy(dtype=float)
    return {
        "roc_auc": roc_auc_score(y, p, sample_weight=w),
        "brier_score": brier_score_loss(y, p, sample_weight=w),
        "log_loss": log_loss(y, p, sample_weight=w, labels=[0, 1]),
        "accuracy": accuracy_score(y, (p >= 0.5).astype(int), sample_weight=w),
        "mean_predicted_probability": float(np.average(p, weights=w)),
        "observed_success_rate": float(np.average(y, weights=w)),
        "checkpoint_rows": len(frame),
        "matches": frame["match_id"].nunique(),
    }


def paired_bootstrap(frame, iterations, rng):
    """Resample matches, preserving paired predictions from both models."""
    match_ids = frame["match_id"].drop_duplicates().to_numpy()
    grouped = {mid: g for mid, g in frame.groupby("match_id", sort=False)}
    metrics = {k: [] for k in ["roc_auc", "brier_score", "log_loss", "accuracy"]}

    for _ in range(iterations):
        sampled = rng.choice(match_ids, size=len(match_ids), replace=True)
        chunks = []
        for draw_id, mid in enumerate(sampled):
            chunk = grouped[mid].copy()
            chunk["_draw_id"] = draw_id
            chunks.append(chunk)
        sample = pd.concat(chunks, ignore_index=True)
        sample["bootstrap_weight"] = (
            1.0 / sample.groupby("_draw_id")["_draw_id"].transform("size")
        )
        y = sample["chase_success"].to_numpy(dtype=int)
        w = sample["bootstrap_weight"].to_numpy(dtype=float)

        try:
            p_a = np.clip(sample["A_baseline"].to_numpy(float), 1e-8, 1 - 1e-8)
            p_c = np.clip(sample["C_shared_gap"].to_numpy(float), 1e-8, 1 - 1e-8)

            vals_a = {
                "roc_auc": roc_auc_score(y, p_a, sample_weight=w),
                "brier_score": brier_score_loss(y, p_a, sample_weight=w),
                "log_loss": log_loss(y, p_a, sample_weight=w, labels=[0, 1]),
                "accuracy": accuracy_score(y, p_a >= 0.5, sample_weight=w),
            }
            vals_c = {
                "roc_auc": roc_auc_score(y, p_c, sample_weight=w),
                "brier_score": brier_score_loss(y, p_c, sample_weight=w),
                "log_loss": log_loss(y, p_c, sample_weight=w, labels=[0, 1]),
                "accuracy": accuracy_score(y, p_c >= 0.5, sample_weight=w),
            }
            for key in metrics:
                metrics[key].append(vals_c[key] - vals_a[key])
        except ValueError:
            continue

    rows = []
    for key, values in metrics.items():
        values = np.asarray(values, dtype=float)
        lower, upper = np.quantile(values, [0.025, 0.975])
        better = np.mean(values > 0) if key in ["roc_auc", "accuracy"] else np.mean(values < 0)
        rows.append({
            "metric": key,
            "C_minus_A_point_difference": (
                metric_values(frame, "C_shared_gap")[key]
                - metric_values(frame, "A_baseline")[key]
            ),
            "bootstrap_iterations": len(values),
            "difference_ci_2_5": lower,
            "difference_ci_97_5": upper,
            "probability_C_better": better,
        })
    return rows


print("\nLoading source datasets...")
balls = pd.read_csv(BALLS_FILE, low_memory=False)
matches = pd.read_csv(MATCHES_FILE, low_memory=False)
competitions = pd.read_csv(COMPETITIONS_FILE, low_memory=False)

for col in ["runs", "extras1_runs", "extras_2_runs", "innings", "is_wicket"]:
    balls[col] = numeric(balls[col]).fillna(0)

matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")
matches["win_team_id_num"] = numeric(matches["win_team_id"])

context = matches.merge(
    competitions[["comp_id", "comp_name", "match_format", "season"]],
    on="comp_id", how="left",
)
name = context["comp_name"].fillna("").str.lower()
fmt = context["match_format"].fillna("").str.lower()
context = context[
    name.str.contains("ipl") | fmt.eq("t20i")
].dropna(subset=["match_id", "match_date", "comp_name"]).copy()
context["competition_group"] = context["comp_name"].apply(competition_group)
context["second_max_balls"] = numeric(
    context["second_inn_revised_overs"]
).fillna(120).astype(int)

balls = balls[balls["match_id"].isin(set(context["match_id"]))].copy()
balls["extras_type_clean"] = (
    balls["extras_type"].fillna("").astype(str).str.strip().str.lower()
)
balls["legal_ball"] = ~balls["extras_type_clean"].isin(["wide", "overstep"])
balls["unresolved_extra"] = balls["extras_type_clean"].isin(["other", "penalty"])
balls["delivery_runs"] = balls["runs"] + balls["extras1_runs"] + balls["extras_2_runs"]

balls = balls.merge(
    context[[
        "match_id", "match_date", "comp_name", "season",
        "win_team_id_num", "second_max_balls",
        "competition_group", "revised_target",
    ]],
    on="match_id", how="inner",
).sort_values(["match_id", "innings", "overid", "over_ball_num"])

print("\nBuilding chase states...")
innings = balls[balls["innings"].isin([1, 2])].copy()

first_outcomes = (
    innings[innings["innings"] == 1]
    .groupby("match_id", as_index=False)
    .agg(
        first_innings_score=("delivery_runs", "sum"),
        first_innings_unresolved=("unresolved_extra", "any"),
    )
)
second_meta = (
    innings[innings["innings"] == 2]
    .groupby("match_id", as_index=False)
    .agg(
        chasing_team_id=("batting_team_id", "first"),
        second_innings_unresolved=("unresolved_extra", "any"),
        second_batting_teams=("batting_team_id", "nunique"),
    )
)
chase_matches = context.merge(first_outcomes, on="match_id", how="inner").merge(
    second_meta, on="match_id", how="inner"
)
chase_matches["chasing_team_id_num"] = numeric(chase_matches["chasing_team_id"])
chase_matches = chase_matches[
    chase_matches["chasing_team_id_num"].notna()
    & chase_matches["win_team_id_num"].notna()
    & (chase_matches["second_batting_teams"] == 1)
    & ~chase_matches["first_innings_unresolved"]
    & ~chase_matches["second_innings_unresolved"]
    & (chase_matches["second_max_balls"] > 0)
].copy()
chase_matches["chase_success"] = (
    chase_matches["chasing_team_id_num"] == chase_matches["win_team_id_num"]
).astype(int)
chase_matches["target"] = numeric(chase_matches["revised_target"]).fillna(
    chase_matches["first_innings_score"] + 1
)
chase_matches = chase_matches[chase_matches["target"] > 0].copy()

second = innings[innings["innings"] == 2].merge(
    chase_matches[[
        "match_id", "target", "chase_success", "second_max_balls",
        "competition_group", "comp_name", "season", "match_date",
    ]],
    on="match_id", how="inner", suffixes=("", "_match"),
).sort_values(["match_id", "overid", "over_ball_num"]).copy()
second = second[~second["unresolved_extra"]].copy()
g = second.groupby("match_id", sort=False)
second["current_score"] = g["delivery_runs"].cumsum() - second["delivery_runs"]
second["wickets_lost"] = g["is_wicket"].cumsum() - second["is_wicket"]
second["legal_balls_bowled"] = (
    g["legal_ball"].cumsum().astype(int) - second["legal_ball"].astype(int)
)
second["legal_balls_remaining"] = (
    second["second_max_balls"] - second["legal_balls_bowled"]
)
second["runs_required"] = second["target"] - second["current_score"]
second = second[
    (second["legal_balls_remaining"] > 0)
    & (second["runs_required"] > 0)
    & (second["wickets_lost"] <= 9)
].copy()
second["required_run_rate"] = (
    second["runs_required"] * 6 / second["legal_balls_remaining"]
)
second["current_run_rate"] = np.where(
    second["legal_balls_bowled"] > 0,
    second["current_score"] * 6 / second["legal_balls_bowled"],
    np.nan,
)
second["run_rate_gap"] = (
    second["required_run_rate"] - second["current_run_rate"]
)
second["balls_band"] = second["legal_balls_remaining"].apply(balls_band)
second = add_interactions(second).replace([np.inf, -np.inf], np.nan)
second = second.dropna(subset=[
    "match_id", "match_date", "comp_name", "wickets_lost",
    "legal_balls_remaining", "required_run_rate", "chase_success",
])
second["chase_success"] = second["chase_success"].astype(int)

print("Usable chase matches:", second["match_id"].nunique())
print("Chase state rows:", len(second))
print("Undefined CRR states:", int(second["current_run_rate"].isna().sum()))

dates = (
    second[["match_id", "match_date"]].drop_duplicates()
    .sort_values(["match_date", "match_id"])
)
splits = {
    "fit": set(dates.loc[dates["match_date"] <= FIT_END, "match_id"]),
    "calibration": set(dates.loc[
        (dates["match_date"] >= CAL_START) & (dates["match_date"] <= CAL_END),
        "match_id"
    ]),
    "test": set(dates.loc[dates["match_date"] >= TEST_START, "match_id"]),
}
for key, ids in splits.items():
    print(f"{key.title()} matches:", len(ids))
    if len(ids) != EXPECTED_MATCHES[key]:
        raise ValueError(
            f"Expected {EXPECTED_MATCHES[key]} {key} matches; found {len(ids)}. "
            "Stop: do not silently change the chronological split."
        )
assert splits["fit"].isdisjoint(splits["calibration"])
assert splits["fit"].isdisjoint(splits["test"])
assert splits["calibration"].isdisjoint(splits["test"])

# As in the prior comparison, use six-ball checkpoints and equal total
# training weight per match. Calibration dates remain untouched.
fit = second[
    second["match_id"].isin(splits["fit"])
    & (second["legal_balls_remaining"] % CHECKPOINT_INTERVAL == 0)
].copy()
calibration = second[
    second["match_id"].isin(splits["calibration"])
    & (second["legal_balls_remaining"] % CHECKPOINT_INTERVAL == 0)
].copy()
test = second[
    second["match_id"].isin(splits["test"])
    & (second["legal_balls_remaining"] % CHECKPOINT_INTERVAL == 0)
].copy()

fit["match_weight"] = equal_match_weights(fit)
test["match_weight"] = equal_match_weights(test)

print("\nApplying the original pressure-state support gate...")
support = (
    fit.groupby(["competition_group", "balls_band", "wickets_lost"])["match_id"]
    .nunique().reset_index(name="support_matches")
)
support_lookup = {
    (r.competition_group, r.balls_band, int(r.wickets_lost)): int(r.support_matches)
    for r in support.itertuples()
}
rrr_ranges = (
    fit.groupby(["competition_group", "balls_band"])["required_run_rate"]
    .quantile([0.01, 0.99]).unstack().reset_index()
    .rename(columns={0.01: "rrr_p01", 0.99: "rrr_p99"})
)
test = test.merge(rrr_ranges, on=["competition_group", "balls_band"], how="left")
test["support_matches"] = [
    support_lookup.get((r.competition_group, r.balls_band, int(r.wickets_lost)), 0)
    for r in test.itertuples()
]
test["supported"] = (
    (test["support_matches"] >= MIN_SUPPORT_MATCHES)
    & (test["required_run_rate"] >= test["rrr_p01"])
    & (test["required_run_rate"] <= test["rrr_p99"])
)

# Explicit, transparent counts by competition and total.
count_rows = []
for scope, frame in [("Overall", test)] + [
    (name, test[test["competition_group"] == name])
    for name in ["IPL", "T20I"]
]:
    count_rows.append({
        "scope": scope,
        "test_matches_with_checkpoint": frame["match_id"].nunique(),
        "test_checkpoint_rows": len(frame),
        "supported_matches": frame.loc[frame["supported"], "match_id"].nunique(),
        "supported_checkpoint_rows": int(frame["supported"].sum()),
        "unsupported_checkpoint_rows": int((~frame["supported"]).sum()),
        "support_coverage_pct": (
            100 * frame["supported"].mean() if len(frame) else np.nan
        ),
    })
pd.DataFrame(count_rows).to_csv(OUTPUT_DIR / "support_coverage.csv", index=False)
print(pd.DataFrame(count_rows).to_string(index=False))

supported_test = test[test["supported"]].copy()
if supported_test.empty or supported_test["chase_success"].nunique() < 2:
    raise ValueError("Supported test states do not contain both outcome classes.")

# Recompute weights after support filtering so each represented match
# contributes equal total evaluation weight within the supported population.
supported_test["match_weight"] = equal_match_weights(supported_test)

print("\nFitting baseline and shared-gap models...")
for model_name, numeric_features in MODELS.items():
    features = numeric_features + CATEGORICAL
    model = build_model(numeric_features)
    model.fit(
        fit[features],
        fit["chase_success"],
        model__sample_weight=fit["match_weight"],
    )
    supported_test[model_name] = model.predict_proba(
        supported_test[features]
    )[:, 1]

# Metrics: overall and by competition, all on the same supported rows.
metric_rows = []
for scope, frame in [("Overall", supported_test)] + [
    (name, supported_test[supported_test["competition_group"] == name])
    for name in ["IPL", "T20I"]
]:
    if len(frame) == 0 or frame["chase_success"].nunique() < 2:
        continue
    for model_name in MODELS:
        metric_rows.append({
            "scope": scope,
            "model": model_name,
            **metric_values(frame, model_name),
        })
metrics = pd.DataFrame(metric_rows)
metrics.to_csv(OUTPUT_DIR / "supported_model_metrics.csv", index=False)
print("\nSupported-state model metrics:")
print(metrics.to_string(index=False))

# Match-level paired bootstrap, using only supported test observations.
rng = np.random.default_rng(RANDOM_SEED)
bootstrap_rows = []
for scope, frame in [("Overall", supported_test)] + [
    (name, supported_test[supported_test["competition_group"] == name])
    for name in ["IPL", "T20I"]
]:
    if len(frame) == 0 or frame["chase_success"].nunique() < 2:
        continue
    for row in paired_bootstrap(frame, BOOTSTRAP_ITERATIONS, rng):
        row["scope"] = scope
        bootstrap_rows.append(row)
bootstrap = pd.DataFrame(bootstrap_rows)
bootstrap.to_csv(OUTPUT_DIR / "supported_paired_bootstrap.csv", index=False)
print("\nPaired match-level bootstrap:")
print(bootstrap.to_string(index=False))

# Save the support rules and chronological split for reproducibility.
pd.DataFrame([
    {"item": "support_match_minimum", "value": MIN_SUPPORT_MATCHES},
    {"item": "rrr_lower_quantile", "value": 0.01},
    {"item": "rrr_upper_quantile", "value": 0.99},
    {"item": "checkpoint_interval_legal_balls", "value": CHECKPOINT_INTERVAL},
    {"item": "fit_end", "value": str(FIT_END.date())},
    {"item": "calibration_start", "value": str(CAL_START.date())},
    {"item": "calibration_end", "value": str(CAL_END.date())},
    {"item": "test_start", "value": str(TEST_START.date())},
]).to_csv(OUTPUT_DIR / "support_rule_and_split.csv", index=False)

print("\nSaved outputs to:", OUTPUT_DIR)
print(" - support_coverage.csv")
print(" - supported_model_metrics.csv")
print(" - supported_paired_bootstrap.csv")
print(" - support_rule_and_split.csv")
print("\nFINAL SUPPORTED CHASE MODEL CHECK COMPLETE")
