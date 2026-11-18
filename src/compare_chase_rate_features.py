from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data Pack"
OUTPUT_DIR = ROOT / "outputs" / "analysis" / "chase_rate_feature_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BALLS_FILE = DATA_DIR / "ball_by_ball.csv"
MATCHES_FILE = DATA_DIR / "matches.csv"
COMPETITIONS_FILE = DATA_DIR / "competitions.csv"

METRICS_FILE = OUTPUT_DIR / "model_metrics.csv"
SUBGROUP_FILE = OUTPUT_DIR / "test_subgroups.csv"
DECILE_FILE = OUTPUT_DIR / "test_probability_deciles.csv"
BOOTSTRAP_FILE = OUTPUT_DIR / "paired_bootstrap_vs_baseline.csv"
SPLIT_FILE = OUTPUT_DIR / "chronological_split.csv"

RANDOM_SEED = 42
CHECKPOINT_INTERVAL = 6
BOOTSTRAP_ITERATIONS = 500

# The previously established three-way chronological evaluation window.
FIT_END = pd.Timestamp("2025-04-07")
CALIBRATION_START = pd.Timestamp("2025-04-08")
CALIBRATION_END = pd.Timestamp("2025-10-31")
TEST_START = pd.Timestamp("2025-11-01")

# Match IDs in the earlier evaluation were 386 fit / 97 calibration / 121 test.
EXPECTED_COUNTS = {
    "fit": 386,
    "calibration": 97,
    "test": 121,
}

MODEL_SPECS = {
    "A_baseline": {
        "numeric": [
            "wickets_lost",
            "legal_balls_remaining",
            "required_run_rate",
            "wickets_x_phase_1_12",
            "wickets_x_phase_13_30",
            "wickets_x_phase_61_90",
            "wickets_x_phase_91_120",
            "wickets_x_t20i",
        ],
    },
    "B_add_crr": {
        "numeric": [
            "wickets_lost",
            "legal_balls_remaining",
            "required_run_rate",
            "current_run_rate",
            "wickets_x_phase_1_12",
            "wickets_x_phase_13_30",
            "wickets_x_phase_61_90",
            "wickets_x_phase_91_120",
            "wickets_x_t20i",
        ],
    },
    "C_add_gap": {
        "numeric": [
            "wickets_lost",
            "legal_balls_remaining",
            "required_run_rate",
            "run_rate_gap",
            "wickets_x_phase_1_12",
            "wickets_x_phase_13_30",
            "wickets_x_phase_61_90",
            "wickets_x_phase_91_120",
            "wickets_x_t20i",
        ],
    },
    "D_add_crr_and_gap": {
        "numeric": [
            "wickets_lost",
            "legal_balls_remaining",
            "required_run_rate",
            "current_run_rate",
            "run_rate_gap",
            "wickets_x_phase_1_12",
            "wickets_x_phase_13_30",
            "wickets_x_phase_61_90",
            "wickets_x_phase_91_120",
            "wickets_x_t20i",
        ],
    },
}

CATEGORICAL_FEATURES = [
    "comp_name",
    "balls_band",
    "competition_group",
]


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
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES,
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


def safe_metrics(y_true, probabilities, weights=None):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-8,
        1 - 1e-8,
    )

    if weights is None:
        weights = np.ones(len(y_true), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    predictions = (probabilities >= 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(
            y_true, probabilities, sample_weight=weights
        ),
        "brier_score": brier_score_loss(
            y_true, probabilities, sample_weight=weights
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
            sample_weight=weights,
            labels=[0, 1],
        ),
        "accuracy": accuracy_score(
            y_true, predictions, sample_weight=weights
        ),
        "mean_predicted_probability": float(
            np.average(probabilities, weights=weights)
        ),
        "observed_success_rate": float(
            np.average(y_true, weights=weights)
        ),
        "observations": int(len(y_true)),
    }


def paired_match_bootstrap_differences(
    frame,
    baseline_col,
    candidate_col,
    iterations,
    rng,
):
    """Resample the same complete matches for both models in every iteration."""
    match_ids = frame["match_id"].drop_duplicates().to_numpy()
    grouped = {
        match_id: group
        for match_id, group in frame.groupby("match_id", sort=False)
    }

    metric_names = ["roc_auc", "brier_score", "log_loss", "accuracy"]
    differences = {name: [] for name in metric_names}

    for _ in range(iterations):
        sampled_ids = rng.choice(
            match_ids,
            size=len(match_ids),
            replace=True,
        )

        pieces = []
        for draw_number, match_id in enumerate(sampled_ids):
            piece = grouped[match_id].copy()
            piece["_bootstrap_match"] = draw_number
            pieces.append(piece)

        sample = pd.concat(pieces, ignore_index=True)
        weights = (
            1.0
            / sample.groupby("_bootstrap_match")[
                "_bootstrap_match"
            ].transform("size")
        )

        try:
            baseline_metrics = safe_metrics(
                sample["chase_success"],
                sample[baseline_col],
                weights,
            )
            candidate_metrics = safe_metrics(
                sample["chase_success"],
                sample[candidate_col],
                weights,
            )

            for metric_name in metric_names:
                differences[metric_name].append(
                    candidate_metrics[metric_name]
                    - baseline_metrics[metric_name]
                )
        except ValueError:
            continue

    return pd.DataFrame(differences)


# ============================================================
# 3. LOAD DATA
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

comp_name_clean = context["comp_name"].fillna("").str.lower()
format_clean = context["match_format"].fillna("").str.lower()

context = context[
    comp_name_clean.str.contains("ipl") | format_clean.eq("t20i")
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
)

balls = balls.sort_values(
    ["match_id", "innings", "overid", "over_ball_num"]
).reset_index(drop=True)


# ============================================================
# 4. REBUILD CHASE OUTCOMES AND PRE-DELIVERY STATES
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

# Retain the existing unresolved-extra treatment.
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

# CRR uses runs and legal balls faced before the current delivery.
# Byes/leg-byes are included in team runs; wides/no-balls do not add legal balls.
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
print("Successful chase matches:", int(
    second.groupby("match_id")["chase_success"].first().sum()
))
print(
    "States with undefined CRR (typically before first legal ball):",
    int(second["current_run_rate"].isna().sum()),
)


# ============================================================
# 5. CHRONOLOGICAL SPLIT
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

split = {
    "fit": fit_ids,
    "calibration": calibration_ids,
    "test": test_ids,
}

for name, ids in split.items():
    print(f"{name.title()} matches:", len(ids))

if len(fit_ids) != EXPECTED_COUNTS["fit"]:
    raise ValueError(
        f"Expected {EXPECTED_COUNTS['fit']} fit matches, "
        f"found {len(fit_ids)}. The source data or date windows differ."
    )
if len(calibration_ids) != EXPECTED_COUNTS["calibration"]:
    raise ValueError(
        f"Expected {EXPECTED_COUNTS['calibration']} calibration matches, "
        f"found {len(calibration_ids)}."
    )
if len(test_ids) != EXPECTED_COUNTS["test"]:
    raise ValueError(
        f"Expected {EXPECTED_COUNTS['test']} test matches, "
        f"found {len(test_ids)}."
    )

assert fit_ids.isdisjoint(calibration_ids)
assert fit_ids.isdisjoint(test_ids)
assert calibration_ids.isdisjoint(test_ids)

split_summary = pd.DataFrame([
    {
        "split": name,
        "matches": len(ids),
        "from_date": match_dates[
            match_dates["match_id"].isin(ids)
        ]["match_date"].min(),
        "to_date": match_dates[
            match_dates["match_id"].isin(ids)
        ]["match_date"].max(),
    }
    for name, ids in split.items()
])
split_summary.to_csv(SPLIT_FILE, index=False)

fit_all = second[second["match_id"].isin(fit_ids)].copy()
calibration_all = second[
    second["match_id"].isin(calibration_ids)
].copy()
test_all = second[second["match_id"].isin(test_ids)].copy()

# Train and evaluate at the same six-ball checkpoints as the established model.
fit = fit_all[
    fit_all["legal_balls_remaining"] % CHECKPOINT_INTERVAL == 0
].copy()
calibration = calibration_all[
    calibration_all["legal_balls_remaining"] % CHECKPOINT_INTERVAL == 0
].copy()
test = test_all[
    test_all["legal_balls_remaining"] % CHECKPOINT_INTERVAL == 0
].copy()

if fit["chase_success"].nunique() != 2:
    raise ValueError("Fit split does not contain both outcome classes.")
if test["chase_success"].nunique() != 2:
    raise ValueError("Test split does not contain both outcome classes.")

fit["match_weight"] = equal_match_weights(fit)
calibration["match_weight"] = equal_match_weights(calibration)
test["match_weight"] = equal_match_weights(test)

print("Fit checkpoint rows:", len(fit))
print("Calibration checkpoint rows:", len(calibration))
print("Test checkpoint rows:", len(test))


# ============================================================
# 6. FIT FOUR MODELS AND EVALUATE
# ============================================================

print("\nFitting four model variants...")

test_predictions = test[
    [
        "match_id",
        "match_date",
        "competition_group",
        "comp_name",
        "balls_band",
        "legal_balls_remaining",
        "wickets_lost",
        "required_run_rate",
        "current_run_rate",
        "run_rate_gap",
        "chase_success",
        "match_weight",
    ]
].copy()

metric_rows = []
fitted_models = {}

for model_name, spec in MODEL_SPECS.items():
    features = spec["numeric"] + CATEGORICAL_FEATURES

    model = build_model(spec["numeric"])

    model.fit(
        fit[features],
        fit["chase_success"],
        model__sample_weight=fit["match_weight"],
    )

    probabilities = model.predict_proba(test[features])[:, 1]
    test_predictions[model_name] = probabilities
    fitted_models[model_name] = model

    metrics = safe_metrics(
        test["chase_success"],
        probabilities,
        test["match_weight"],
    )

    metric_rows.append({
        "model": model_name,
        "numeric_features": ", ".join(spec["numeric"]),
        **metrics,
    })

metrics_frame = pd.DataFrame(metric_rows)
metrics_frame.to_csv(METRICS_FILE, index=False)

print("\nTest metrics:")
print(metrics_frame.to_string(index=False))


# ============================================================
# 7. SUBGROUP PERFORMANCE
# ============================================================

print("\nCalculating competition and phase subgroup performance...")

subgroup_rows = []

for group_column in ["competition_group", "balls_band"]:
    for group_value, group in test_predictions.groupby(
        group_column, dropna=False
    ):
        if group["chase_success"].nunique() < 2:
            continue

        for model_name in MODEL_SPECS:
            metrics = safe_metrics(
                group["chase_success"],
                group[model_name],
                group["match_weight"],
            )

            subgroup_rows.append({
                "group_type": group_column,
                "group": group_value,
                "model": model_name,
                **metrics,
            })

subgroups = pd.DataFrame(subgroup_rows)
subgroups.to_csv(SUBGROUP_FILE, index=False)

print(subgroups.to_string(index=False))


# ============================================================
# 8. TEST PROBABILITY DECILES
# ============================================================

print("\nBuilding test probability-decile diagnostics...")

decile_rows = []

for model_name in MODEL_SPECS:
    frame = test_predictions[
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


# ============================================================
# 9. PAIRED MATCH-LEVEL BOOTSTRAP VS BASELINE
# ============================================================

print("\nRunning paired match-level bootstrap...")

rng = np.random.default_rng(RANDOM_SEED)
bootstrap_rows = []

baseline = "A_baseline"

for candidate in [
    "B_add_crr",
    "C_add_gap",
    "D_add_crr_and_gap",
]:
    paired = test_predictions[
        [
            "match_id",
            "chase_success",
            "match_weight",
            baseline,
            candidate,
        ]
    ].copy()

    paired_boot = paired_match_bootstrap_differences(
        paired,
        baseline,
        candidate,
        BOOTSTRAP_ITERATIONS,
        rng,
    )

    for metric in [
        "roc_auc",
        "brier_score",
        "log_loss",
        "accuracy",
    ]:
        base_point = safe_metrics(
            paired["chase_success"],
            paired[baseline],
            paired["match_weight"],
        )[metric]
        candidate_point = safe_metrics(
            paired["chase_success"],
            paired[candidate],
            paired["match_weight"],
        )[metric]

        # Differences are paired because both models use the same
        # resampled match IDs within each bootstrap iteration.
        differences = paired_boot[metric].to_numpy()
        count = len(differences)

        bootstrap_rows.append({
            "comparison": f"{candidate} minus {baseline}",
            "metric": metric,
            "baseline_point": base_point,
            "candidate_point": candidate_point,
            "point_difference": candidate_point - base_point,
            "bootstrap_iterations": count,
            "difference_ci_2_5": (
                float(np.quantile(differences, 0.025))
                if count else np.nan
            ),
            "difference_ci_97_5": (
                float(np.quantile(differences, 0.975))
                if count else np.nan
            ),
            "probability_candidate_better": (
                float(np.mean(differences > 0))
                if metric in ["roc_auc", "accuracy"] and count
                else (
                    float(np.mean(differences < 0))
                    if count else np.nan
                )
            ),
        })

bootstrap = pd.DataFrame(bootstrap_rows)
bootstrap.to_csv(BOOTSTRAP_FILE, index=False)


# ============================================================
# 10. FINAL DIAGNOSTICS
# ============================================================

print("\nPaired bootstrap versus baseline:")
print(bootstrap.to_string(index=False))

print("\nTest probability deciles:")
print(deciles.to_string(index=False))

print("\nCRR / RRR gap diagnostics:")
print(
    test_predictions[
        [
            "required_run_rate",
            "current_run_rate",
            "run_rate_gap",
        ]
    ].describe().to_string()
)

print("\nFiles saved:")
print(METRICS_FILE)
print(SUBGROUP_FILE)
print(DECILE_FILE)
print(BOOTSTRAP_FILE)
print(SPLIT_FILE)

print("\n" + "=" * 78)
print("CHASE RATE FEATURE COMPARISON COMPLETE")
print("=" * 78)
