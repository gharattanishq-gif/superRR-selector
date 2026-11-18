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
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data Pack"
OUTPUT_DIR = ROOT / "outputs" / "analysis" / "competition_specific_gap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BALLS_FILE = DATA_DIR / "ball_by_ball.csv"
MATCHES_FILE = DATA_DIR / "matches.csv"
COMPETITIONS_FILE = DATA_DIR / "competitions.csv"

METRICS_FILE = OUTPUT_DIR / "model_metrics.csv"
SUBGROUP_FILE = OUTPUT_DIR / "competition_subgroup_metrics.csv"
BOOTSTRAP_FILE = OUTPUT_DIR / "paired_bootstrap_comparisons.csv"
DECILE_FILE = OUTPUT_DIR / "probability_deciles.csv"
SPLIT_FILE = OUTPUT_DIR / "chronological_split.csv"

RANDOM_SEED = 42
CHECKPOINT_INTERVAL = 6
BOOTSTRAP_ITERATIONS = 500

FIT_END = pd.Timestamp("2025-04-07")
CALIBRATION_START = pd.Timestamp("2025-04-08")
CALIBRATION_END = pd.Timestamp("2025-10-31")
TEST_START = pd.Timestamp("2025-11-01")

EXPECTED_SPLIT_COUNTS = {
    "fit": 386,
    "calibration": 97,
    "test": 121,
}

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

CATEGORICAL = [
    "comp_name",
    "balls_band",
    "competition_group",
]

MODEL_FEATURES = {
    "A_baseline": BASE_NUMERIC,
    "C_shared_gap": BASE_NUMERIC + ["run_rate_gap"],
    "E_competition_specific_gap": (
        BASE_NUMERIC + ["run_rate_gap", "run_rate_gap_x_t20i"]
    ),
}


# ============================================================
# 2. HELPERS
# ============================================================

def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def clean_extra_type(series):
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


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
        data["wickets_lost"]
        * (data["balls_band"] == "1-12").astype(int)
    )
    data["wickets_x_phase_13_30"] = (
        data["wickets_lost"]
        * (data["balls_band"] == "13-30").astype(int)
    )
    data["wickets_x_phase_61_90"] = (
        data["wickets_lost"]
        * (data["balls_band"] == "61-90").astype(int)
    )
    data["wickets_x_phase_91_120"] = (
        data["wickets_lost"]
        * (data["balls_band"] == "91-120").astype(int)
    )
    data["wickets_x_t20i"] = (
        data["wickets_lost"]
        * (data["competition_group"] == "T20I").astype(int)
    )

    # IPL is the reference group. This interaction allows the RRR-CRR
    # gap slope to differ in T20Is while retaining a shared base slope.
    data["run_rate_gap_x_t20i"] = (
        data["run_rate_gap"]
        * (data["competition_group"] == "T20I").astype(int)
    )

    return data


def equal_match_weights(data, match_col="match_id"):
    counts = data.groupby(match_col)[match_col].transform("size")
    return 1.0 / counts


def build_model(numeric_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL,
        ),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_SEED,
            ),
        ),
    ])


def calculate_metrics(frame, probability_col):
    y = frame["chase_success"].to_numpy(dtype=int)
    p = np.clip(
        frame[probability_col].to_numpy(dtype=float),
        1e-8,
        1 - 1e-8,
    )
    weights = frame["match_weight"].to_numpy(dtype=float)

    return {
        "roc_auc": roc_auc_score(y, p, sample_weight=weights),
        "brier_score": brier_score_loss(y, p, sample_weight=weights),
        "log_loss": log_loss(
            y, p, sample_weight=weights, labels=[0, 1]
        ),
        "accuracy": accuracy_score(
            y, (p >= 0.5).astype(int), sample_weight=weights
        ),
        "mean_predicted_probability": float(
            np.average(p, weights=weights)
        ),
        "observed_success_rate": float(
            np.average(y, weights=weights)
        ),
        "observations": len(frame),
        "matches": frame["match_id"].nunique(),
    }


def bootstrap_metric_difference(
    frame,
    baseline_col,
    candidate_col,
    group_column,
    group_value,
    iterations,
    rng,
):
    """Paired bootstrap: both models see the same resampled match draws."""
    subset = frame.copy()

    if group_column is not None:
        subset = subset[subset[group_column] == group_value].copy()

    match_ids = subset["match_id"].drop_duplicates().to_numpy()
    grouped = {
        match_id: group
        for match_id, group in subset.groupby("match_id", sort=False)
    }

    metric_names = ["roc_auc", "brier_score", "log_loss", "accuracy"]
    differences = {metric: [] for metric in metric_names}

    for _ in range(iterations):
        sampled_ids = rng.choice(
            match_ids,
            size=len(match_ids),
            replace=True,
        )

        parts = []
        for draw_number, match_id in enumerate(sampled_ids):
            part = grouped[match_id].copy()
            part["_bootstrap_match"] = draw_number
            parts.append(part)

        sample = pd.concat(parts, ignore_index=True)
        sample["bootstrap_weight"] = (
            1.0
            / sample.groupby("_bootstrap_match")[
                "_bootstrap_match"
            ].transform("size")
        )

        y = sample["chase_success"].to_numpy(dtype=int)
        weights = sample["bootstrap_weight"].to_numpy(dtype=float)

        try:
            model_metrics = {}
            for model_col in [baseline_col, candidate_col]:
                p = np.clip(
                    sample[model_col].to_numpy(dtype=float),
                    1e-8,
                    1 - 1e-8,
                )
                model_metrics[model_col] = {
                    "roc_auc": roc_auc_score(
                        y, p, sample_weight=weights
                    ),
                    "brier_score": brier_score_loss(
                        y, p, sample_weight=weights
                    ),
                    "log_loss": log_loss(
                        y, p, sample_weight=weights, labels=[0, 1]
                    ),
                    "accuracy": accuracy_score(
                        y,
                        (p >= 0.5).astype(int),
                        sample_weight=weights,
                    ),
                }

            for metric in metric_names:
                differences[metric].append(
                    model_metrics[candidate_col][metric]
                    - model_metrics[baseline_col][metric]
                )
        except ValueError:
            continue

    return differences


# ============================================================
# 3. LOAD AND PROCESS SOURCE DATA
# ============================================================

print("\nLoading source datasets...")

balls = pd.read_csv(BALLS_FILE, low_memory=False)
matches = pd.read_csv(MATCHES_FILE, low_memory=False)
competitions = pd.read_csv(COMPETITIONS_FILE, low_memory=False)

for column in [
    "runs",
    "extras1_runs",
    "extras_2_runs",
    "innings",
    "overid",
    "over_ball_num",
    "is_wicket",
]:
    balls[column] = numeric(balls[column]).fillna(0)

matches["match_date"] = pd.to_datetime(
    matches["match_date"], errors="coerce"
)
matches["win_team_id_num"] = numeric(matches["win_team_id"])

context = matches.merge(
    competitions[
        ["comp_id", "comp_name", "match_format", "season"]
    ],
    on="comp_id",
    how="left",
)

comp_clean = context["comp_name"].fillna("").str.lower()
format_clean = context["match_format"].fillna("").str.lower()

context = context[
    comp_clean.str.contains("ipl") | format_clean.eq("t20i")
].dropna(
    subset=["match_id", "match_date", "comp_name"]
).copy()

context["competition_group"] = (
    context["comp_name"].apply(competition_group)
)
context["second_max_balls"] = (
    numeric(context["second_inn_revised_overs"])
    .fillna(120)
    .astype(int)
)

balls = balls[
    balls["match_id"].isin(set(context["match_id"]))
].copy()

balls["extras_type_clean"] = clean_extra_type(balls["extras_type"])
balls["legal_ball"] = ~balls["extras_type_clean"].isin(
    ["wide", "overstep"]
)
balls["unresolved_extra"] = balls["extras_type_clean"].isin(
    ["other", "penalty"]
)
balls["delivery_runs"] = (
    balls["runs"] + balls["extras1_runs"] + balls["extras_2_runs"]
)

balls = balls.merge(
    context[
        [
            "match_id",
            "match_date",
            "comp_name",
            "season",
            "win_team_id_num",
            "second_max_balls",
            "competition_group",
            "revised_target",
        ]
    ],
    on="match_id",
    how="inner",
).sort_values(
    ["match_id", "innings", "overid", "over_ball_num"]
).reset_index(drop=True)


# ============================================================
# 4. BUILD CHASE OUTCOMES AND PRE-DELIVERY FEATURES
# ============================================================

print("\nBuilding chase states...")

innings_balls = balls[balls["innings"].isin([1, 2])].copy()

first_outcomes = (
    innings_balls[innings_balls["innings"] == 1]
    .groupby("match_id", as_index=False)
    .agg(
        first_innings_score=("delivery_runs", "sum"),
        first_innings_unresolved=("unresolved_extra", "any"),
    )
)

second_metadata = (
    innings_balls[innings_balls["innings"] == 2]
    .groupby("match_id", as_index=False)
    .agg(
        chasing_team_id=("batting_team_id", "first"),
        second_innings_unresolved=("unresolved_extra", "any"),
        second_batting_teams=("batting_team_id", "nunique"),
    )
)

chase_matches = (
    context.merge(first_outcomes, on="match_id", how="inner")
    .merge(second_metadata, on="match_id", how="inner")
)

chase_matches["chasing_team_id_num"] = numeric(
    chase_matches["chasing_team_id"]
)

chase_matches = chase_matches[
    chase_matches["chasing_team_id_num"].notna()
    & chase_matches["win_team_id_num"].notna()
    & (chase_matches["second_batting_teams"] == 1)
    & ~chase_matches["first_innings_unresolved"]
    & ~chase_matches["second_innings_unresolved"]
    & (chase_matches["second_max_balls"] > 0)
].copy()

chase_matches["chase_success"] = (
    chase_matches["chasing_team_id_num"]
    == chase_matches["win_team_id_num"]
).astype(int)

chase_matches["target"] = numeric(chase_matches["revised_target"])
chase_matches["target"] = chase_matches["target"].fillna(
    chase_matches["first_innings_score"] + 1
)
chase_matches = chase_matches[chase_matches["target"] > 0].copy()

second = (
    innings_balls[innings_balls["innings"] == 2]
    .merge(
        chase_matches[
            [
                "match_id",
                "target",
                "chase_success",
                "second_max_balls",
                "competition_group",
                "comp_name",
                "season",
                "match_date",
            ]
        ],
        on="match_id",
        how="inner",
        suffixes=("", "_match"),
    )
    .sort_values(["match_id", "overid", "over_ball_num"])
    .copy()
)

second = second[~second["unresolved_extra"]].copy()
grouped = second.groupby("match_id", sort=False)

second["current_score"] = (
    grouped["delivery_runs"].cumsum() - second["delivery_runs"]
)
second["wickets_lost"] = (
    grouped["is_wicket"].cumsum() - second["is_wicket"]
)
second["legal_balls_bowled"] = (
    grouped["legal_ball"].cumsum().astype(int)
    - second["legal_ball"].astype(int)
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
    second["runs_required"] * 6
    / second["legal_balls_remaining"]
)
second["current_run_rate"] = np.where(
    second["legal_balls_bowled"] > 0,
    second["current_score"] * 6 / second["legal_balls_bowled"],
    np.nan,
)
second["run_rate_gap"] = (
    second["required_run_rate"] - second["current_run_rate"]
)
second["balls_band"] = second["legal_balls_remaining"].apply(
    balls_band
)
second = add_interactions(second)
second = second.replace([np.inf, -np.inf], np.nan)
second = second.dropna(
    subset=[
        "match_id",
        "match_date",
        "comp_name",
        "wickets_lost",
        "legal_balls_remaining",
        "required_run_rate",
        "chase_success",
    ]
)
second["chase_success"] = second["chase_success"].astype(int)

print("Usable chase matches:", second["match_id"].nunique())
print("Chase state rows:", len(second))
print(
    "Undefined CRR states:",
    int(second["current_run_rate"].isna().sum()),
)


# ============================================================
# 5. RECREATE THE SAME CHRONOLOGICAL SPLIT
# ============================================================

print("\nCreating chronological fit / calibration / test split...")

match_dates = (
    second[["match_id", "match_date"]]
    .drop_duplicates()
    .sort_values(["match_date", "match_id"])
)

fit_ids = set(
    match_dates.loc[
        match_dates["match_date"] <= FIT_END, "match_id"
    ]
)
calibration_ids = set(
    match_dates.loc[
        (match_dates["match_date"] >= CALIBRATION_START)
        & (match_dates["match_date"] <= CALIBRATION_END),
        "match_id",
    ]
)
test_ids = set(
    match_dates.loc[
        match_dates["match_date"] >= TEST_START, "match_id"
    ]
)

split_ids = {
    "fit": fit_ids,
    "calibration": calibration_ids,
    "test": test_ids,
}

for split_name, ids in split_ids.items():
    print(f"{split_name.title()} matches:", len(ids))
    if len(ids) != EXPECTED_SPLIT_COUNTS[split_name]:
        raise ValueError(
            f"{split_name} split expected "
            f"{EXPECTED_SPLIT_COUNTS[split_name]} matches, got {len(ids)}."
        )

assert fit_ids.isdisjoint(calibration_ids)
assert fit_ids.isdisjoint(test_ids)
assert calibration_ids.isdisjoint(test_ids)

split_rows = []
for split_name, ids in split_ids.items():
    dates = match_dates.loc[
        match_dates["match_id"].isin(ids), "match_date"
    ]
    split_rows.append({
        "split": split_name,
        "matches": len(ids),
        "from_date": dates.min(),
        "to_date": dates.max(),
    })
pd.DataFrame(split_rows).to_csv(SPLIT_FILE, index=False)

fit = second[
    second["match_id"].isin(fit_ids)
    & (second["legal_balls_remaining"] % CHECKPOINT_INTERVAL == 0)
].copy()
calibration = second[
    second["match_id"].isin(calibration_ids)
    & (second["legal_balls_remaining"] % CHECKPOINT_INTERVAL == 0)
].copy()
test = second[
    second["match_id"].isin(test_ids)
    & (second["legal_balls_remaining"] % CHECKPOINT_INTERVAL == 0)
].copy()

if fit["chase_success"].nunique() < 2:
    raise ValueError("Fit checkpoint sample lacks both outcome classes.")
if test["chase_success"].nunique() < 2:
    raise ValueError("Test checkpoint sample lacks both outcome classes.")

fit["match_weight"] = equal_match_weights(fit)
calibration["match_weight"] = equal_match_weights(calibration)
test["match_weight"] = equal_match_weights(test)

print("Fit checkpoint rows:", len(fit))
print("Calibration checkpoint rows:", len(calibration))
print("Test checkpoint rows:", len(test))


# ============================================================
# 6. FIT BASELINE, SHARED-GAP AND COMPETITION-SPECIFIC-GAP
# ============================================================

print("\nFitting three candidate models...")

predictions = test[
    [
        "match_id",
        "match_date",
        "competition_group",
        "comp_name",
        "balls_band",
        "chase_success",
        "match_weight",
        "required_run_rate",
        "current_run_rate",
        "run_rate_gap",
    ]
].copy()

metric_rows = []
models = {}

for model_name, numeric_features in MODEL_FEATURES.items():
    feature_columns = numeric_features + CATEGORICAL
    model = build_model(numeric_features)

    model.fit(
        fit[feature_columns],
        fit["chase_success"],
        model__sample_weight=fit["match_weight"],
    )

    predictions[model_name] = model.predict_proba(
        test[feature_columns]
    )[:, 1]

    models[model_name] = model
    metric_rows.append({
        "model": model_name,
        "numeric_features": ", ".join(numeric_features),
        **calculate_metrics(predictions, model_name),
    })

metrics = pd.DataFrame(metric_rows)
metrics.to_csv(METRICS_FILE, index=False)

print("\nOverall test metrics:")
print(metrics.to_string(index=False))


# ============================================================
# 7. COMPETITION-SPECIFIC METRICS
# ============================================================

print("\nCompetition subgroup metrics...")

subgroup_rows = []

for competition, group in predictions.groupby("competition_group"):
    if group["chase_success"].nunique() < 2:
        continue

    for model_name in MODEL_FEATURES:
        subgroup_rows.append({
            "competition_group": competition,
            "model": model_name,
            **calculate_metrics(group, model_name),
        })

subgroups = pd.DataFrame(subgroup_rows)
subgroups.to_csv(SUBGROUP_FILE, index=False)
print(subgroups.to_string(index=False))


# ============================================================
# 8. PAIRED MATCH-LEVEL BOOTSTRAP
# ============================================================

print("\nRunning paired match-level bootstrap...")

rng = np.random.default_rng(RANDOM_SEED)
comparisons = [
    ("C_shared_gap", "A_baseline"),
    ("E_competition_specific_gap", "A_baseline"),
    ("E_competition_specific_gap", "C_shared_gap"),
]

bootstrap_rows = []

for candidate, baseline in comparisons:
    for scope_name, group_column, group_value in [
        ("Overall", None, None),
        ("IPL", "competition_group", "IPL"),
        ("T20I", "competition_group", "T20I"),
    ]:
        differences = bootstrap_metric_difference(
            predictions,
            baseline,
            candidate,
            group_column,
            group_value,
            BOOTSTRAP_ITERATIONS,
            rng,
        )

        scope = predictions
        if group_column is not None:
            scope = scope[scope[group_column] == group_value]

        baseline_point = calculate_metrics(scope, baseline)
        candidate_point = calculate_metrics(scope, candidate)

        for metric_name, values in differences.items():
            values = np.asarray(values, dtype=float)
            if len(values) == 0:
                lower = upper = probability_better = np.nan
            else:
                lower, upper = np.quantile(values, [0.025, 0.975])
                # Higher is better for AUC/accuracy; lower is better for losses.
                probability_better = (
                    np.mean(values > 0)
                    if metric_name in ["roc_auc", "accuracy"]
                    else np.mean(values < 0)
                )

            bootstrap_rows.append({
                "scope": scope_name,
                "comparison": f"{candidate} minus {baseline}",
                "metric": metric_name,
                "baseline_point": baseline_point[metric_name],
                "candidate_point": candidate_point[metric_name],
                "point_difference": (
                    candidate_point[metric_name]
                    - baseline_point[metric_name]
                ),
                "bootstrap_iterations": len(values),
                "difference_ci_2_5": lower,
                "difference_ci_97_5": upper,
                "probability_candidate_better": probability_better,
            })

bootstrap = pd.DataFrame(bootstrap_rows)
bootstrap.to_csv(BOOTSTRAP_FILE, index=False)

print("\nPaired bootstrap results:")
print(bootstrap.to_string(index=False))


# ============================================================
# 9. PROBABILITY DECILE DIAGNOSTICS
# ============================================================

print("\nProbability-decile diagnostics...")

decile_rows = []

for model_name in MODEL_FEATURES:
    frame = predictions[
        ["match_id", "chase_success", "match_weight", model_name]
    ].copy()

    frame["decile"] = pd.qcut(
        frame[model_name].rank(method="first"),
        q=10,
        labels=False,
        duplicates="drop",
    ) + 1

    for decile, group in frame.groupby("decile"):
        decile_rows.append({
            "model": model_name,
            "decile": int(decile),
            "rows": len(group),
            "matches": group["match_id"].nunique(),
            "mean_predicted": float(
                np.average(
                    group[model_name],
                    weights=group["match_weight"],
                )
            ),
            "observed_success": float(
                np.average(
                    group["chase_success"],
                    weights=group["match_weight"],
                )
            ),
        })

deciles = pd.DataFrame(decile_rows)
deciles.to_csv(DECILE_FILE, index=False)
print(deciles.to_string(index=False))


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\nFiles saved:")
print(METRICS_FILE)
print(SUBGROUP_FILE)
print(BOOTSTRAP_FILE)
print(DECILE_FILE)
print(SPLIT_FILE)

print("\n" + "=" * 78)
print("COMPETITION-SPECIFIC GAP MODEL COMPARISON COMPLETE")
print("=" * 78)
