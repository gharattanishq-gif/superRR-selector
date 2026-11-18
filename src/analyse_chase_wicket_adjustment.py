
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
)


# ============================================================
# PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "Data Pack"
OUTPUT_DIR = ROOT / "outputs" / "analysis" / "chase_adjustment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BALLS_FILE = DATA_DIR / "ball_by_ball.csv"
MATCHES_FILE = DATA_DIR / "matches.csv"
COMPETITIONS_FILE = DATA_DIR / "competitions.csv"

RANDOM_STATE = 42
CHECKPOINT_INTERVAL = 6

# Chronological split:
# 80% of matches for development, 20% for final testing.
OUTER_TRAIN_FRACTION = 0.80

# Within development data:
# 80% for fitting the base model, 20% for calibration.
MODEL_FIT_FRACTION = 0.80


# ============================================================
# HELPERS
# ============================================================

def numeric_id(series):
    return pd.to_numeric(series, errors="coerce")


def build_preprocessor():
    numeric_features = [
        "wickets_lost",
        "legal_balls_remaining",
        "required_run_rate",
    ]

    categorical_features = ["comp_name"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "competition",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    return preprocessor


def make_base_model():
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight=None,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def weighted_calibration_table(df, probability_column, bins=10):
    result = df.copy()

    result["prediction_band"] = pd.qcut(
        result[probability_column],
        q=bins,
        duplicates="drop",
    )

    rows = []

    for band, group in result.groupby(
        "prediction_band",
        observed=True,
    ):
        weights = group["evaluation_weight"]

        rows.append({
            "prediction_band": str(band),
            "states": len(group),
            "matches": group["match_id"].nunique(),
            "mean_predicted_probability": np.average(
                group[probability_column],
                weights=weights,
            ),
            "observed_chase_success_rate": np.average(
                group["chase_success"],
                weights=weights,
            ),
        })

    return pd.DataFrame(rows)


def evaluate_model(test, probability_column, model_name):
    y_true = test["chase_success"]
    probabilities = test[probability_column]
    weights = test["evaluation_weight"]

    return {
        "model": model_name,
        "test_matches": test["match_id"].nunique(),
        "test_states": len(test),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
            sample_weight=weights,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
            sample_weight=weights,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
            sample_weight=weights,
            labels=[0, 1],
        ),
        "accuracy_at_0_5": accuracy_score(
            y_true,
            (probabilities >= 0.5).astype(int),
            sample_weight=weights,
        ),
        "mean_predicted_probability": np.average(
            probabilities,
            weights=weights,
        ),
        "observed_success_rate": np.average(
            y_true,
            weights=weights,
        ),
    }


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

balls = pd.read_csv(BALLS_FILE, low_memory=False)
matches = pd.read_csv(MATCHES_FILE, low_memory=False)
competitions = pd.read_csv(COMPETITIONS_FILE, low_memory=False)

for col in [
    "runs",
    "extras1_runs",
    "extras_2_runs",
    "innings",
    "overid",
    "over_ball_num",
    "is_wicket",
]:
    balls[col] = pd.to_numeric(
        balls[col],
        errors="coerce",
    ).fillna(0)

matches["match_date"] = pd.to_datetime(
    matches["match_date"],
    errors="coerce",
)

matches["win_team_id_num"] = numeric_id(
    matches["win_team_id"]
)

context = matches.merge(
    competitions[["comp_id", "comp_name", "match_format", "season"]],
    on="comp_id",
    how="left",
)

competition_name = context["comp_name"].fillna("").str.lower()
match_format = context["match_format"].fillna("").str.lower()

context["is_primary_competition"] = (
    competition_name.str.contains("ipl")
    | match_format.eq("t20i")
)

context = context[
    context["is_primary_competition"]
].copy()

context = context.dropna(
    subset=["match_id", "match_date"]
)

context["second_max_balls"] = (
    pd.to_numeric(
        context["second_inn_revised_overs"],
        errors="coerce",
    )
    .fillna(120)
    .astype(int)
)


# ============================================================
# PREPARE MATCH OUTCOMES
# ============================================================

print("Preparing match outcomes...")

primary_match_ids = set(context["match_id"])

balls = balls[
    balls["match_id"].isin(primary_match_ids)
].copy()

balls["extras_type_clean"] = (
    balls["extras_type"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

balls["legal_ball"] = ~balls["extras_type_clean"].isin(
    ["wide", "overstep"]
)

balls["unresolved_extra"] = balls[
    "extras_type_clean"
].isin(["other", "penalty"])

balls["delivery_runs"] = (
    balls["runs"]
    + balls["extras1_runs"]
    + balls["extras_2_runs"]
)

balls["is_wicket"] = balls["is_wicket"].astype(int)

innings_balls = balls[
    balls["innings"].isin([1, 2])
].copy()

first_innings = (
    innings_balls[innings_balls["innings"] == 1]
    .groupby("match_id", as_index=False)
    .agg(
        first_innings_score=("delivery_runs", "sum"),
        first_innings_has_unresolved=("unresolved_extra", "any"),
    )
)

second_metadata = (
    innings_balls[innings_balls["innings"] == 2]
    .groupby("match_id", as_index=False)
    .agg(
        chasing_team_id=("batting_team_id", "first"),
        second_innings_has_unresolved=("unresolved_extra", "any"),
        second_innings_batting_teams=("batting_team_id", "nunique"),
    )
)

match_data = (
    context.merge(first_innings, on="match_id", how="inner")
    .merge(second_metadata, on="match_id", how="inner")
)

match_data["chasing_team_id_num"] = numeric_id(
    match_data["chasing_team_id"]
)

match_data = match_data[
    match_data["chasing_team_id_num"].notna()
    & match_data["win_team_id_num"].notna()
    & (match_data["second_innings_batting_teams"] == 1)
].copy()

match_data["chase_success"] = (
    match_data["chasing_team_id_num"]
    == match_data["win_team_id_num"]
).astype(int)

match_data["target"] = pd.to_numeric(
    match_data["revised_target"],
    errors="coerce",
)

match_data["target"] = match_data["target"].fillna(
    match_data["first_innings_score"] + 1
)

match_data = match_data[
    ~match_data["first_innings_has_unresolved"]
    & ~match_data["second_innings_has_unresolved"]
    & (match_data["target"] > 0)
    & (match_data["second_max_balls"] > 0)
].copy()

print("\n--- MATCH OUTCOME VALIDATION ---")
print(f"Usable matches: {len(match_data)}")
print(
    "Successful chases:",
    int(match_data["chase_success"].sum()),
)
print(
    "Unsuccessful chases:",
    int((match_data["chase_success"] == 0).sum()),
)

if match_data["chase_success"].nunique() != 2:
    raise ValueError("Match outcomes contain only one class.")


# ============================================================
# BUILD SECOND-INNINGS STATES
# ============================================================

print("\nBuilding second-innings states...")

match_columns = [
    "match_id",
    "match_date",
    "comp_name",
    "season",
    "target",
    "second_max_balls",
    "chase_success",
]

second = innings_balls[
    innings_balls["innings"] == 2
].merge(
    match_data[match_columns],
    on="match_id",
    how="inner",
)

second = second.sort_values(
    ["match_id", "overid", "over_ball_num"]
).copy()

second = second[
    ~second["unresolved_extra"]
].copy()

grouped = second.groupby("match_id", sort=False)

second["current_score"] = (
    grouped["delivery_runs"].cumsum()
    - second["delivery_runs"]
)

second["wickets_lost"] = (
    grouped["is_wicket"].cumsum()
    - second["is_wicket"]
)

second["legal_balls_bowled"] = (
    grouped["legal_ball"].cumsum().astype(int)
    - second["legal_ball"].astype(int)
)

second["legal_balls_remaining"] = (
    second["second_max_balls"]
    - second["legal_balls_bowled"]
)

second["runs_required"] = (
    second["target"] - second["current_score"]
)

second = second[
    (second["legal_balls_remaining"] > 0)
    & (second["runs_required"] > 0)
    & (second["wickets_lost"] <= 9)
].copy()

second["required_run_rate"] = (
    second["runs_required"] * 6
    / second["legal_balls_remaining"]
)

# Retain one pre-delivery checkpoint every six legal balls.
second = second[
    second["legal_balls_remaining"] % CHECKPOINT_INTERVAL == 0
].copy()

model_data = second[
    [
        "match_id",
        "match_date",
        "comp_name",
        "season",
        "wickets_lost",
        "legal_balls_remaining",
        "runs_required",
        "required_run_rate",
        "chase_success",
    ]
].copy()

model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan,
)

model_data = model_data.dropna(
    subset=[
        "match_id",
        "match_date",
        "wickets_lost",
        "legal_balls_remaining",
        "required_run_rate",
        "chase_success",
        "comp_name",
    ]
)

model_data["chase_success"] = model_data[
    "chase_success"
].astype(int)

print(f"Checkpoint observations: {len(model_data):,}")
print(f"Matches represented: {model_data['match_id'].nunique():,}")

if model_data["match_id"].nunique() < 30:
    raise ValueError("Insufficient matches for this experiment.")


# ============================================================
# CHRONOLOGICAL OUTER SPLIT
# ============================================================

match_dates = (
    model_data[["match_id", "match_date"]]
    .drop_duplicates("match_id")
    .sort_values(["match_date", "match_id"])
)

outer_split = int(
    len(match_dates) * OUTER_TRAIN_FRACTION
)

outer_split = min(
    max(outer_split, 2),
    len(match_dates) - 1,
)

development_matches = match_dates.iloc[:outer_split]
test_matches = match_dates.iloc[outer_split:]

development_ids = set(development_matches["match_id"])
test_ids = set(test_matches["match_id"])

development = model_data[
    model_data["match_id"].isin(development_ids)
].copy()

test = model_data[
    model_data["match_id"].isin(test_ids)
].copy()

# Equal total weight per match.
development["match_weight"] = (
    1 / development.groupby("match_id")["match_id"].transform("size")
)

test["evaluation_weight"] = (
    1 / test.groupby("match_id")["match_id"].transform("size")
)


# ============================================================
# CHRONOLOGICAL INNER SPLIT
# ============================================================

development_dates = (
    development[["match_id", "match_date"]]
    .drop_duplicates("match_id")
    .sort_values(["match_date", "match_id"])
)

inner_split = int(
    len(development_dates) * MODEL_FIT_FRACTION
)

inner_split = min(
    max(inner_split, 1),
    len(development_dates) - 1,
)

fit_matches = development_dates.iloc[:inner_split]
calibration_matches = development_dates.iloc[inner_split:]

fit_ids = set(fit_matches["match_id"])
calibration_ids = set(calibration_matches["match_id"])

fit_data = development[
    development["match_id"].isin(fit_ids)
].copy()

calibration_data = development[
    development["match_id"].isin(calibration_ids)
].copy()

# Equal total weight per match within each split.
fit_data["match_weight"] = (
    1 / fit_data.groupby("match_id")["match_id"].transform("size")
)

calibration_data["match_weight"] = (
    1 / calibration_data.groupby("match_id")["match_id"].transform("size")
)

print("\n--- CHRONOLOGICAL SPLITS ---")
print(f"Model-fitting matches: {fit_data['match_id'].nunique()}")
print(f"Calibration matches: {calibration_data['match_id'].nunique()}")
print(f"Final test matches: {test['match_id'].nunique()}")
print(f"Model fitting through: {fit_data['match_date'].max().date()}")
print(f"Calibration from: {calibration_data['match_date'].min().date()}")
print(f"Final test from: {test['match_date'].min().date()}")

for name, dataset in [
    ("Model-fitting", fit_data),
    ("Calibration", calibration_data),
    ("Final test", test),
]:
    if dataset["chase_success"].nunique() != 2:
        raise ValueError(
            f"{name} split does not contain both outcome classes."
        )


# ============================================================
# TRAIN BASE LOGISTIC REGRESSION
# ============================================================

features = [
    "wickets_lost",
    "legal_balls_remaining",
    "required_run_rate",
    "comp_name",
]

print("\nTraining base logistic regression...")

base_model = make_base_model()

base_model.fit(
    fit_data[features],
    fit_data["chase_success"],
    model__sample_weight=fit_data["match_weight"],
)

# Probabilities on the later calibration period.
calibration_raw = base_model.predict_proba(
    calibration_data[features]
)[:, 1]

# Probabilities on the untouched final test period.
test_raw = base_model.predict_proba(
    test[features]
)[:, 1]

calibration_data["raw_probability"] = calibration_raw
test["original_probability"] = test_raw


# ============================================================
# PLATT / SIGMOID CALIBRATION
# ============================================================

print("Fitting Platt calibration...")

# Platt scaling fits a logistic regression to the log-odds
# of the base model's predicted probabilities.

EPSILON = 1e-6

calibration_logits = np.log(
    np.clip(calibration_raw, EPSILON, 1 - EPSILON)
    / (
        1 - np.clip(calibration_raw, EPSILON, 1 - EPSILON)
    )
).reshape(-1, 1)

test_logits = np.log(
    np.clip(test_raw, EPSILON, 1 - EPSILON)
    / (
        1 - np.clip(test_raw, EPSILON, 1 - EPSILON)
    )
).reshape(-1, 1)

platt_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE,
)

platt_model.fit(
    calibration_logits,
    calibration_data["chase_success"],
    sample_weight=calibration_data["match_weight"],
)

test["platt_probability"] = platt_model.predict_proba(
    test_logits
)[:, 1]


# ============================================================
# ISOTONIC CALIBRATION
# ============================================================

print("Fitting isotonic calibration...")

isotonic_model = IsotonicRegression(
    y_min=0,
    y_max=1,
    out_of_bounds="clip",
)

isotonic_model.fit(
    calibration_raw,
    calibration_data["chase_success"],
    sample_weight=calibration_data["match_weight"],
)

test["isotonic_probability"] = isotonic_model.predict(
    test_raw
)


# ============================================================
# EVALUATE ALL THREE APPROACHES
# ============================================================

print("\nEvaluating models on untouched final test set...")

results = []

probability_columns = {
    "Original logistic regression": "original_probability",
    "Platt calibrated": "platt_probability",
    "Isotonic calibrated": "isotonic_probability",
}

for model_name, probability_column in probability_columns.items():
    results.append(
        evaluate_model(
            test,
            probability_column,
            model_name,
        )
    )

metrics_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("CALIBRATION COMPARISON")
print("=" * 80)
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ============================================================
# CALIBRATION TABLES
# ============================================================

calibration_tables = []

for model_name, probability_column in probability_columns.items():
    table = weighted_calibration_table(
        test,
        probability_column,
        bins=10,
    )

    table.insert(0, "model", model_name)
    calibration_tables.append(table)

all_calibration = pd.concat(
    calibration_tables,
    ignore_index=True,
)

print("\n" + "=" * 80)
print("CALIBRATION TABLES")
print("=" * 80)

for model_name in probability_columns:
    print(f"\n--- {model_name} ---")

    print(
        all_calibration[
            all_calibration["model"] == model_name
        ].drop(columns="model").to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

metrics_df.to_csv(
    OUTPUT_DIR / "chase_calibration_model_comparison.csv",
    index=False,
)

all_calibration.to_csv(
    OUTPUT_DIR / "chase_calibration_tables.csv",
    index=False,
)

test[
    [
        "match_id",
        "match_date",
        "comp_name",
        "season",
        "wickets_lost",
        "legal_balls_remaining",
        "runs_required",
        "required_run_rate",
        "chase_success",
        "original_probability",
        "platt_probability",
        "isotonic_probability",
        "evaluation_weight",
    ]
].to_csv(
    OUTPUT_DIR / "chase_calibration_test_predictions.csv",
    index=False,
)

print("\n" + "=" * 80)
print(f"Outputs saved to: {OUTPUT_DIR}")
print("=" * 80)
