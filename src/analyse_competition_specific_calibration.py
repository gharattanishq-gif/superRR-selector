
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
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
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "Data Pack"
OUTPUT_DIR = ROOT / "outputs" / "analysis" / "chase_adjustment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BALLS_FILE = DATA_DIR / "ball_by_ball.csv"
MATCHES_FILE = DATA_DIR / "matches.csv"
COMPETITIONS_FILE = DATA_DIR / "competitions.csv"

OUTER_TRAIN_FRACTION = 0.80
MODEL_FIT_FRACTION = 0.80
CHECKPOINT_INTERVAL = 6

RANDOM_SEED = 42

BASE_NUMERIC = [
    "wickets_lost",
    "legal_balls_remaining",
    "required_run_rate",
]

BASE_CATEGORICAL = ["comp_name"]

INTERACTION_NUMERIC = BASE_NUMERIC + [
    "wickets_x_phase_1_12",
    "wickets_x_phase_13_30",
    "wickets_x_phase_61_90",
    "wickets_x_phase_91_120",
    "wickets_x_t20i",
]

INTERACTION_CATEGORICAL = [
    "comp_name",
    "balls_band",
    "competition_group",
]


# ============================================================
# HELPERS
# ============================================================

def numeric_id(series):
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


def add_interaction_features(data):
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


def equal_match_weights(data, match_column):
    counts = data.groupby(match_column)[match_column].transform("size")
    return 1 / counts


def build_interaction_model():

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                INTERACTION_NUMERIC,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                INTERACTION_CATEGORICAL,
            ),
        ]
    )

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


def clip_probabilities(probabilities):
    return np.clip(
        np.asarray(probabilities, dtype=float),
        1e-6,
        1 - 1e-6,
    )


def probability_logits(probabilities):
    probabilities = clip_probabilities(probabilities)
    return np.log(probabilities / (1 - probabilities))


def calculate_metrics(y_true, probabilities, weights):

    probabilities = clip_probabilities(probabilities)
    predictions = (probabilities >= 0.5).astype(int)

    return {
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
        "accuracy": accuracy_score(
            y_true,
            predictions,
            sample_weight=weights,
        ),
        "mean_predicted_success": np.average(
            probabilities,
            weights=weights,
        ),
        "observed_success": np.average(
            y_true,
            weights=weights,
        ),
    }


def fit_global_platt(probabilities, outcomes, weights):

    logits = probability_logits(probabilities).reshape(-1, 1)

    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=2000,
    )

    model.fit(
        logits,
        outcomes,
        sample_weight=weights,
    )

    return model


def apply_global_platt(model, probabilities):

    logits = probability_logits(probabilities).reshape(-1, 1)

    return model.predict_proba(logits)[:, 1]


def fit_competition_intercept_calibrator(
    probabilities,
    outcomes,
    competition_groups,
    weights,
):
    """
    Shared slope, separate competition intercepts.

    Reference category: T20I.

    logit(P) = intercept
               + shared_slope * logit(raw_probability)
               + IPL_intercept_adjustment
    """

    logits = probability_logits(probabilities)

    ipl_indicator = (
        pd.Series(competition_groups)
        .reset_index(drop=True)
        .eq("IPL")
        .astype(int)
        .to_numpy()
    )

    features = np.column_stack([
        logits,
        ipl_indicator,
    ])

    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=2000,
    )

    calibrator.fit(
        features,
        outcomes,
        sample_weight=weights,
    )

    return calibrator


def apply_competition_intercept_calibrator(
    calibrator,
    probabilities,
    competition_groups,
):

    logits = probability_logits(probabilities)

    ipl_indicator = (
        pd.Series(competition_groups)
        .reset_index(drop=True)
        .eq("IPL")
        .astype(int)
        .to_numpy()
    )

    features = np.column_stack([
        logits,
        ipl_indicator,
    ])

    return calibrator.predict_proba(features)[:, 1]


def calibration_diagnostics(y, probabilities, weights):

    logits = probability_logits(probabilities).reshape(-1, 1)

    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=2000,
    )

    model.fit(
        logits,
        y,
        sample_weight=weights,
    )

    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0][0]),
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


# ============================================================
# COMPETITION CONTEXT
# ============================================================

context = matches.merge(
    competitions[
        ["comp_id", "comp_name", "match_format", "season"]
    ],
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
# DELIVERY PROCESSING
# ============================================================

balls = balls[
    balls["match_id"].isin(set(context["match_id"]))
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


# ============================================================
# MATCH OUTCOMES
# ============================================================

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
    context.merge(
        first_innings,
        on="match_id",
        how="inner",
    )
    .merge(
        second_metadata,
        on="match_id",
        how="inner",
    )
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

print("\n--- MATCH VALIDATION ---")
print(f"Usable matches: {len(match_data)}")
print(f"Successful chases: {int(match_data['chase_success'].sum())}")
print(
    "Unsuccessful chases:",
    int((match_data["chase_success"] == 0).sum()),
)


# ============================================================
# BUILD SECOND-INNINGS STATES
# ============================================================

print("\nBuilding second-innings states...")

match_columns = [
    "match_id",
    "match_date",
    "comp_name",
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

second = second[
    second["legal_balls_remaining"] % CHECKPOINT_INTERVAL == 0
].copy()

model_data = second[
    [
        "match_id",
        "match_date",
        "comp_name",
        "wickets_lost",
        "legal_balls_remaining",
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
        "comp_name",
        "wickets_lost",
        "legal_balls_remaining",
        "required_run_rate",
        "chase_success",
    ]
)

model_data["chase_success"] = model_data[
    "chase_success"
].astype(int)

model_data["competition_group"] = (
    model_data["comp_name"].apply(competition_group)
)

model_data["balls_band"] = (
    model_data["legal_balls_remaining"].apply(balls_band)
)

model_data = add_interaction_features(model_data)

print(f"Checkpoint observations: {len(model_data):,}")
print(f"Matches represented: {model_data['match_id'].nunique()}")


# ============================================================
# CHRONOLOGICAL THREE-WAY SPLIT
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
    max(outer_split, 1),
    len(match_dates) - 1,
)

development_ids = match_dates.iloc[
    :outer_split
]["match_id"].tolist()

test_ids = set(
    match_dates.iloc[outer_split:]["match_id"]
)

inner_split = int(
    len(development_ids) * MODEL_FIT_FRACTION
)

inner_split = min(
    max(inner_split, 1),
    len(development_ids) - 1,
)

fit_ids = set(development_ids[:inner_split])
calibration_ids = set(development_ids[inner_split:])

fit_data = model_data[
    model_data["match_id"].isin(fit_ids)
].copy()

calibration_data = model_data[
    model_data["match_id"].isin(calibration_ids)
].copy()

test_data = model_data[
    model_data["match_id"].isin(test_ids)
].copy()

print("\n--- CHRONOLOGICAL THREE-WAY SPLIT ---")
print(f"Model-fitting matches: {fit_data['match_id'].nunique()}")
print(f"Calibration matches: {calibration_data['match_id'].nunique()}")
print(f"Final test matches: {test_data['match_id'].nunique()}")

print(f"Model-fitting through: {fit_data['match_date'].max().date()}")
print(
    "Calibration period:",
    calibration_data["match_date"].min().date(),
    "to",
    calibration_data["match_date"].max().date(),
)
print(f"Final test from: {test_data['match_date'].min().date()}")


# ============================================================
# EQUAL-MATCH WEIGHTS
# ============================================================

for dataset in [fit_data, calibration_data, test_data]:
    dataset["match_weight"] = equal_match_weights(
        dataset,
        "match_id",
    )


# ============================================================
# COMPETITION SAMPLE COUNTS
# ============================================================

def competition_sample_summary(data, label):

    summary = (
        data.groupby("competition_group")
        .agg(
            checkpoint_rows=("chase_success", "size"),
            matches=("match_id", "nunique"),
            successful_checkpoints=("chase_success", "sum"),
        )
        .reset_index()
    )

    # Equal-match weighted observed success rate.
    rates = (
        data.groupby("competition_group")
        .apply(
            lambda group: np.average(
                group["chase_success"],
                weights=group["match_weight"],
            ),
            include_groups=False,
        )
        .rename("observed_success_rate")
        .reset_index()
    )

    summary = summary.merge(
        rates,
        on="competition_group",
        how="left",
    )

    summary["dataset"] = label

    return summary


sample_summary = pd.concat([
    competition_sample_summary(fit_data, "Model fitting"),
    competition_sample_summary(calibration_data, "Calibration"),
    competition_sample_summary(test_data, "Final test"),
], ignore_index=True)

print("\n--- COMPETITION SAMPLE COUNTS ---")
print(sample_summary.to_string(index=False))


# ============================================================
# FIT INTERACTION MODEL
# ============================================================

print("\nFitting interaction model...")

interaction_model = build_interaction_model()

interaction_model.fit(
    fit_data[
        INTERACTION_NUMERIC + INTERACTION_CATEGORICAL
    ],
    fit_data["chase_success"],
    model__sample_weight=fit_data["match_weight"],
)

calibration_raw = interaction_model.predict_proba(
    calibration_data[
        INTERACTION_NUMERIC + INTERACTION_CATEGORICAL
    ]
)[:, 1]

test_raw = interaction_model.predict_proba(
    test_data[
        INTERACTION_NUMERIC + INTERACTION_CATEGORICAL
    ]
)[:, 1]


# ============================================================
# FIT CALIBRATORS ON CALIBRATION DATA ONLY
# ============================================================

print("\nFitting calibration methods...")

calibration_y = calibration_data["chase_success"].to_numpy()
calibration_weights = calibration_data["match_weight"].to_numpy()
calibration_competitions = (
    calibration_data["competition_group"].to_numpy()
)

# Global Platt.
global_platt = fit_global_platt(
    calibration_raw,
    calibration_y,
    calibration_weights,
)

# Competition-specific intercept, shared slope.
competition_platt = fit_competition_intercept_calibrator(
    calibration_raw,
    calibration_y,
    calibration_competitions,
    calibration_weights,
)

# Isotonic retained as a reference.
isotonic = IsotonicRegression(
    out_of_bounds="clip",
)

isotonic.fit(
    calibration_raw,
    calibration_y,
    sample_weight=calibration_weights,
)


# ============================================================
# APPLY CALIBRATORS TO UNTOUCHED TEST DATA
# ============================================================

test_competitions = test_data[
    "competition_group"
].to_numpy()

predictions = {
    "Raw": clip_probabilities(test_raw),

    "Global Platt": clip_probabilities(
        apply_global_platt(
            global_platt,
            test_raw,
        )
    ),

    "Competition intercept Platt": clip_probabilities(
        apply_competition_intercept_calibrator(
            competition_platt,
            test_raw,
            test_competitions,
        )
    ),

    "Isotonic": clip_probabilities(
        isotonic.predict(test_raw)
    ),
}


# ============================================================
# REPORT CALIBRATION PARAMETERS
# ============================================================

global_slope = float(global_platt.coef_[0][0])
global_intercept = float(global_platt.intercept_[0])

shared_slope = float(competition_platt.coef_[0][0])
t20i_intercept = float(competition_platt.intercept_[0])
ipl_intercept_adjustment = float(competition_platt.coef_[0][1])

print("\n--- CALIBRATION PARAMETERS ---")
print(f"Global Platt intercept: {global_intercept:.6f}")
print(f"Global Platt slope: {global_slope:.6f}")
print(f"Competition model shared slope: {shared_slope:.6f}")
print(f"T20I intercept: {t20i_intercept:.6f}")
print(f"IPL intercept adjustment: {ipl_intercept_adjustment:.6f}")
print(
    "IPL total intercept:",
    f"{t20i_intercept + ipl_intercept_adjustment:.6f}",
)


# ============================================================
# OVERALL TEST PERFORMANCE
# ============================================================

test_y = test_data["chase_success"].to_numpy()
test_weights = test_data["match_weight"].to_numpy()

result_rows = []

for method_name, probabilities in predictions.items():

    metrics = calculate_metrics(
        test_y,
        probabilities,
        test_weights,
    )

    diagnostics = {
        "calibration_intercept": np.nan,
        "calibration_slope": np.nan,
    }

    if method_name != "Isotonic":
        diagnostics = calibration_diagnostics(
            test_y,
            probabilities,
            test_weights,
        )

    result_rows.append({
        "model": "Interaction",
        "calibration_method": method_name,
        **metrics,
        **diagnostics,
    })

overall_summary = pd.DataFrame(result_rows)

print("\n--- FINAL TEST CALIBRATION RESULTS ---")
print(
    overall_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# COMPETITION-SPECIFIC TEST RESULTS
# ============================================================

competition_rows = []

for method_name, probabilities in predictions.items():

    subgroup = test_data[
        [
            "match_id",
            "competition_group",
            "chase_success",
            "match_weight",
        ]
    ].copy()

    subgroup["predicted_probability"] = probabilities

    for comp_group, group in subgroup.groupby(
        "competition_group"
    ):

        metrics = calculate_metrics(
            group["chase_success"],
            group["predicted_probability"],
            group["match_weight"],
        )

        competition_rows.append({
            "calibration_method": method_name,
            "competition_group": comp_group,
            "matches": group["match_id"].nunique(),
            **metrics,
        })

competition_summary = pd.DataFrame(competition_rows)

print("\n--- COMPETITION-SPECIFIC TEST RESULTS ---")
print(
    competition_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# CALIBRATION DECILES
# ============================================================

decile_rows = []

for method_name, probabilities in predictions.items():

    table = test_data[
        [
            "match_id",
            "chase_success",
            "match_weight",
        ]
    ].copy()

    table["predicted_probability"] = probabilities

    table["probability_decile"] = pd.qcut(
        table["predicted_probability"].rank(
            method="first"
        ),
        q=10,
        labels=False,
        duplicates="drop",
    ) + 1

    for decile, group in table.groupby(
        "probability_decile",
        observed=True,
    ):

        decile_rows.append({
            "calibration_method": method_name,
            "probability_decile": int(decile),
            "checkpoint_rows": len(group),
            "distinct_matches": group["match_id"].nunique(),
            "mean_predicted_probability": np.average(
                group["predicted_probability"],
                weights=group["match_weight"],
            ),
            "observed_success_rate": np.average(
                group["chase_success"],
                weights=group["match_weight"],
            ),
        })

decile_summary = pd.DataFrame(decile_rows)

print("\n--- CALIBRATION BY PROBABILITY DECILE ---")
print(
    decile_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

overall_summary.to_csv(
    OUTPUT_DIR / "competition_calibration_comparison.csv",
    index=False,
)

competition_summary.to_csv(
    OUTPUT_DIR / "competition_calibration_subgroups.csv",
    index=False,
)

decile_summary.to_csv(
    OUTPUT_DIR / "competition_calibration_deciles.csv",
    index=False,
)

sample_summary.to_csv(
    OUTPUT_DIR / "competition_calibration_sample_counts.csv",
    index=False,
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 95)
print("COMPETITION-SPECIFIC CALIBRATION ANALYSIS COMPLETE")
print("=" * 95)

print(f"Model-fitting matches: {fit_data['match_id'].nunique()}")
print(f"Calibration matches: {calibration_data['match_id'].nunique()}")
print(f"Final test matches: {test_data['match_id'].nunique()}")

print(f"\nOutputs saved to: {OUTPUT_DIR}")
print("=" * 95)
