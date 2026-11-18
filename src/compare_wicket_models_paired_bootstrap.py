
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
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "Data Pack"
OUTPUT_DIR = ROOT / "outputs" / "analysis" / "chase_adjustment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BALLS_FILE = DATA_DIR / "ball_by_ball.csv"
MATCHES_FILE = DATA_DIR / "matches.csv"
COMPETITIONS_FILE = DATA_DIR / "competitions.csv"

TRAIN_FRACTION = 0.80
CHECKPOINT_INTERVAL = 6

BOOTSTRAP_ITERATIONS = 500
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


def build_model(interaction=False):

    if interaction:
        numeric_features = INTERACTION_NUMERIC
        categorical_features = INTERACTION_CATEGORICAL
    else:
        numeric_features = BASE_NUMERIC
        categorical_features = BASE_CATEGORICAL

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
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
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


def calculate_metrics(y_true, probabilities, weights):

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

# Same provisional treatment as previous chase analyses.
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
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

match_dates = (
    model_data[["match_id", "match_date"]]
    .drop_duplicates("match_id")
    .sort_values(["match_date", "match_id"])
)

split_position = int(
    len(match_dates) * TRAIN_FRACTION
)

split_position = min(
    max(split_position, 1),
    len(match_dates) - 1,
)

train_ids = set(
    match_dates.iloc[:split_position]["match_id"]
)

test_ids = set(
    match_dates.iloc[split_position:]["match_id"]
)

train = model_data[
    model_data["match_id"].isin(train_ids)
].copy()

test = model_data[
    model_data["match_id"].isin(test_ids)
].copy()

print("\n--- CHRONOLOGICAL SPLIT ---")
print(f"Training matches: {train['match_id'].nunique()}")
print(f"Testing matches: {test['match_id'].nunique()}")
print(f"Training through: {train['match_date'].max().date()}")
print(f"Testing from: {test['match_date'].min().date()}")


# ============================================================
# ORIGINAL TEST-SET COMPARISON
# ============================================================

print("\nFitting original models...")

train["match_weight"] = equal_match_weights(
    train,
    "match_id",
)

test["match_weight"] = equal_match_weights(
    test,
    "match_id",
)

baseline_model = build_model(interaction=False)

baseline_model.fit(
    train[BASE_NUMERIC + BASE_CATEGORICAL],
    train["chase_success"],
    model__sample_weight=train["match_weight"],
)

interaction_model = build_model(interaction=True)

interaction_model.fit(
    train[INTERACTION_NUMERIC + INTERACTION_CATEGORICAL],
    train["chase_success"],
    model__sample_weight=train["match_weight"],
)

baseline_probabilities = baseline_model.predict_proba(
    test[BASE_NUMERIC + BASE_CATEGORICAL]
)[:, 1]

interaction_probabilities = interaction_model.predict_proba(
    test[INTERACTION_NUMERIC + INTERACTION_CATEGORICAL]
)[:, 1]

original_baseline_metrics = calculate_metrics(
    test["chase_success"],
    baseline_probabilities,
    test["match_weight"],
)

original_interaction_metrics = calculate_metrics(
    test["chase_success"],
    interaction_probabilities,
    test["match_weight"],
)

print("\n--- ORIGINAL TEST-SET PERFORMANCE ---")

for metric in original_baseline_metrics:
    print(
        f"{metric}: "
        f"Baseline={original_baseline_metrics[metric]:.4f} | "
        f"Interaction={original_interaction_metrics[metric]:.4f}"
    )


# ============================================================
# PREPARE MATCH-LEVEL RESAMPLING
# ============================================================

train_match_ids = train["match_id"].unique()
test_match_ids = test["match_id"].unique()

train_by_match = {
    match_id: group.copy()
    for match_id, group in train.groupby("match_id")
}

test_by_match = {
    match_id: group.copy()
    for match_id, group in test.groupby("match_id")
}


def resample_matches(match_ids, match_lookup, rng, id_column):

    sampled_ids = rng.choice(
        match_ids,
        size=len(match_ids),
        replace=True,
    )

    frames = []

    for draw_number, match_id in enumerate(sampled_ids):

        frame = match_lookup[match_id].copy()

        # A repeated source match is a separate bootstrap cluster.
        frame[id_column] = draw_number

        frames.append(frame)

    sample = pd.concat(
        frames,
        ignore_index=True,
    )

    sample["bootstrap_weight"] = equal_match_weights(
        sample,
        id_column,
    )

    return sample


# ============================================================
# PAIRED MATCH-LEVEL BOOTSTRAP
# ============================================================

print("\nStarting paired match-level bootstrap...")
print(f"Iterations: {BOOTSTRAP_ITERATIONS}")

rng = np.random.default_rng(RANDOM_SEED)

bootstrap_rows = []
failed_iterations = 0

for iteration in range(BOOTSTRAP_ITERATIONS):

    try:

        # Draw the training and test samples ONCE.
        # Both models receive these exact same samples.
        bootstrap_train = resample_matches(
            train_match_ids,
            train_by_match,
            rng,
            "bootstrap_train_match_id",
        )

        bootstrap_test = resample_matches(
            test_match_ids,
            test_by_match,
            rng,
            "bootstrap_test_match_id",
        )

        # Fit baseline model.
        baseline = build_model(interaction=False)

        baseline.fit(
            bootstrap_train[
                BASE_NUMERIC + BASE_CATEGORICAL
            ],
            bootstrap_train["chase_success"],
            model__sample_weight=bootstrap_train[
                "bootstrap_weight"
            ],
        )

        # Fit interaction model on the SAME training sample.
        interaction = build_model(interaction=True)

        interaction.fit(
            bootstrap_train[
                INTERACTION_NUMERIC + INTERACTION_CATEGORICAL
            ],
            bootstrap_train["chase_success"],
            model__sample_weight=bootstrap_train[
                "bootstrap_weight"
            ],
        )

        # Evaluate both on the SAME test sample.
        baseline_probs = baseline.predict_proba(
            bootstrap_test[
                BASE_NUMERIC + BASE_CATEGORICAL
            ]
        )[:, 1]

        interaction_probs = interaction.predict_proba(
            bootstrap_test[
                INTERACTION_NUMERIC + INTERACTION_CATEGORICAL
            ]
        )[:, 1]

        y_true = bootstrap_test["chase_success"]
        weights = bootstrap_test["bootstrap_weight"]

        # AUC cannot be calculated if a bootstrap test sample
        # contains only one outcome class.
        if y_true.nunique() < 2:
            raise ValueError(
                "Bootstrap test sample contains only one outcome class."
            )

        baseline_metrics = calculate_metrics(
            y_true,
            baseline_probs,
            weights,
        )

        interaction_metrics = calculate_metrics(
            y_true,
            interaction_probs,
            weights,
        )

        row = {
            "iteration": iteration + 1,
        }

        for metric in baseline_metrics:

            row[f"baseline_{metric}"] = baseline_metrics[metric]
            row[f"interaction_{metric}"] = interaction_metrics[metric]

            row[f"delta_{metric}"] = (
                interaction_metrics[metric]
                - baseline_metrics[metric]
            )

        bootstrap_rows.append(row)

    except Exception as error:

        failed_iterations += 1

        print(
            f"Iteration {iteration + 1} failed: {error}"
        )

    if (iteration + 1) % 25 == 0:
        print(
            f"Completed {iteration + 1}/{BOOTSTRAP_ITERATIONS} "
            f"| Failed: {failed_iterations}"
        )


# ============================================================
# BOOTSTRAP COMPLETION
# ============================================================

bootstrap_df = pd.DataFrame(bootstrap_rows)

if bootstrap_df.empty:
    raise ValueError(
        "All bootstrap iterations failed. No results were produced."
    )

successful_iterations = bootstrap_df[
    "iteration"
].nunique()

print("\n--- BOOTSTRAP COMPLETION ---")
print(f"Requested iterations: {BOOTSTRAP_ITERATIONS}")
print(f"Successful iterations: {successful_iterations}")
print(f"Failed iterations: {failed_iterations}")


# ============================================================
# SUMMARISE PAIRED DIFFERENCES
# ============================================================

metrics = [
    "roc_auc",
    "brier_score",
    "log_loss",
    "accuracy",
    "mean_predicted_success",
]

summary_rows = []

for metric in metrics:

    baseline_column = f"baseline_{metric}"
    interaction_column = f"interaction_{metric}"
    delta_column = f"delta_{metric}"

    baseline_values = bootstrap_df[
        baseline_column
    ]

    interaction_values = bootstrap_df[
        interaction_column
    ]

    delta_values = bootstrap_df[
        delta_column
    ]

    summary_rows.append({
        "metric": metric,

        "original_baseline": original_baseline_metrics[metric],
        "original_interaction": original_interaction_metrics[metric],
        "original_difference": (
            original_interaction_metrics[metric]
            - original_baseline_metrics[metric]
        ),

        "bootstrap_mean_baseline": baseline_values.mean(),
        "bootstrap_mean_interaction": interaction_values.mean(),

        "mean_paired_difference": delta_values.mean(),
        "difference_ci_lower": delta_values.quantile(0.025),
        "difference_ci_upper": delta_values.quantile(0.975),

        "bootstrap_probability_interaction_better": (
            (
                delta_values > 0
                if metric in ["roc_auc", "accuracy"]
                else delta_values < 0
            ).mean()
        ),
    })

summary = pd.DataFrame(summary_rows)


# ============================================================
# INTERPRETATION FLAGS
# ============================================================

def improvement_interpretation(row):

    metric = row["metric"]

    lower = row["difference_ci_lower"]
    upper = row["difference_ci_upper"]

    if metric in ["roc_auc", "accuracy"]:
        if lower > 0:
            return "Interaction model significantly better"
        if upper < 0:
            return "Baseline model significantly better"
        return "No clear difference"

    if metric in ["brier_score", "log_loss"]:
        if upper < 0:
            return "Interaction model significantly better"
        if lower > 0:
            return "Baseline model significantly better"
        return "No clear difference"

    return "Descriptive only"


summary["interpretation"] = summary.apply(
    improvement_interpretation,
    axis=1,
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

bootstrap_df.to_csv(
    OUTPUT_DIR / "wicket_paired_bootstrap_all_iterations.csv",
    index=False,
)

summary.to_csv(
    OUTPUT_DIR / "wicket_paired_bootstrap_summary.csv",
    index=False,
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 100)
print("PAIRED MATCH-LEVEL BOOTSTRAP: MODEL COMPARISON")
print("=" * 100)

print("\nDifference is calculated as INTERACTION minus BASELINE.")
print("For ROC-AUC and accuracy, positive is better.")
print("For Brier score and log loss, negative is better.")

print("\n--- PAIRED BOOTSTRAP CONFIDENCE INTERVALS ---")

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\n--- FINAL INTERPRETATION ---")

for _, row in summary.iterrows():
    print(
        f"{row['metric']}: {row['interpretation']} "
        f"| Probability interaction model is better: "
        f"{row['bootstrap_probability_interaction_better']:.1%}"
    )

print("\n" + "=" * 100)
print(f"Outputs saved to: {OUTPUT_DIR}")
print("=" * 100)
