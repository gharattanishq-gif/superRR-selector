
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
MIN_SUPPORT_MATCHES = 5

BOOTSTRAP_ITERATIONS = 500
RANDOM_SEED = 42

PHASES = ["1-12", "13-30", "31-60", "61-90", "91-120"]
REFERENCE_PHASE = "31-60"

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


def equal_match_weights(data, match_column):
    counts = data.groupby(match_column)[match_column].transform("size")
    return 1 / counts


def weighted_metrics(y_true, probabilities, weights):

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

# Same provisional delivery treatment as previous analyses.
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
# FIT AND COMPARE MODELS
# ============================================================

print("\nFitting baseline and interaction models...")

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

baseline_metrics = weighted_metrics(
    test["chase_success"],
    baseline_probabilities,
    test["match_weight"],
)

interaction_metrics = weighted_metrics(
    test["chase_success"],
    interaction_probabilities,
    test["match_weight"],
)

comparison = pd.DataFrame([
    {
        "model": "Baseline additive logistic",
        **baseline_metrics,
    },
    {
        "model": "Phase + competition interactions",
        **interaction_metrics,
    },
])

print("\n--- CHRONOLOGICAL TEST PERFORMANCE ---")
print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

comparison.to_csv(
    OUTPUT_DIR / "wicket_interaction_model_comparison.csv",
    index=False,
)


# ============================================================
# HISTORICAL SUPPORT AND FIXED COMPARISON RANGES
# ============================================================

support = (
    train.groupby(
        [
            "competition_group",
            "balls_band",
            "wickets_lost",
        ]
    )["match_id"]
    .nunique()
    .reset_index(name="support_matches")
)

support_lookup = {
    (
        row["competition_group"],
        row["balls_band"],
        int(row["wickets_lost"]),
    ): int(row["support_matches"])
    for _, row in support.iterrows()
}

test_groups = (
    test[
        ["competition_group", "balls_band"]
    ]
    .drop_duplicates()
    .sort_values(
        ["competition_group", "balls_band"]
    )
)

supported_groups = []

for _, row in test_groups.iterrows():

    comp_group = row["competition_group"]
    ball_group = row["balls_band"]

    maximum = -1

    for wicket_count in range(10):

        count = support_lookup.get(
            (comp_group, ball_group, wicket_count),
            0,
        )

        if count >= MIN_SUPPORT_MATCHES:
            maximum = wicket_count
        else:
            break

    if maximum >= 0:
        supported_groups.append({
            "competition_group": comp_group,
            "balls_band": ball_group,
            "max_supported_wickets": maximum,
        })

supported_groups = pd.DataFrame(supported_groups)

print("\n--- SUPPORTED COMPARISON GROUPS ---")
print(supported_groups.to_string(index=False))

supported_groups.to_csv(
    OUTPUT_DIR / "wicket_interaction_supported_groups.csv",
    index=False,
)


# ============================================================
# MATCH-LEVEL BOOTSTRAP PREPARATION
# ============================================================

train_match_ids = train["match_id"].unique()

train_by_match = {
    match_id: group.copy()
    for match_id, group in train.groupby("match_id")
}

test_by_group = {}

for _, group_row in supported_groups.iterrows():

    comp_group = group_row["competition_group"]
    ball_group = group_row["balls_band"]

    subset = test[
        (test["competition_group"] == comp_group)
        & (test["balls_band"] == ball_group)
    ].copy()

    test_by_group[(comp_group, ball_group)] = {
        match_id: group.copy()
        for match_id, group in subset.groupby("match_id")
    }


def bootstrap_training_sample(rng):

    sampled_ids = rng.choice(
        train_match_ids,
        size=len(train_match_ids),
        replace=True,
    )

    frames = []

    for draw_number, match_id in enumerate(sampled_ids):

        frame = train_by_match[match_id].copy()
        frame["bootstrap_match_id"] = draw_number

        frames.append(frame)

    sample = pd.concat(
        frames,
        ignore_index=True,
    )

    sample["bootstrap_weight"] = equal_match_weights(
        sample,
        "bootstrap_match_id",
    )

    return sample


def bootstrap_test_sample(match_dict, rng):

    original_ids = np.array(list(match_dict.keys()))

    sampled_ids = rng.choice(
        original_ids,
        size=len(original_ids),
        replace=True,
    )

    frames = []

    for draw_number, match_id in enumerate(sampled_ids):

        frame = match_dict[match_id].copy()
        frame["bootstrap_test_match_id"] = draw_number

        frames.append(frame)

    sample = pd.concat(
        frames,
        ignore_index=True,
    )

    sample["bootstrap_weight"] = equal_match_weights(
        sample,
        "bootstrap_test_match_id",
    )

    return sample


# ============================================================
# BOOTSTRAP INTERACTION-MODEL MARGINAL EFFECTS
# ============================================================

print("\nStarting interaction-model match-level bootstrap...")
print(f"Iterations: {BOOTSTRAP_ITERATIONS}")

rng = np.random.default_rng(RANDOM_SEED)

bootstrap_results = []
failed_iterations = 0

for iteration in range(BOOTSTRAP_ITERATIONS):

    try:

        bootstrap_train = bootstrap_training_sample(rng)

        model = build_model(interaction=True)

        model.fit(
            bootstrap_train[
                INTERACTION_NUMERIC + INTERACTION_CATEGORICAL
            ],
            bootstrap_train["chase_success"],
            model__sample_weight=bootstrap_train[
                "bootstrap_weight"
            ],
        )

        for _, group_row in supported_groups.iterrows():

            comp_group = group_row["competition_group"]
            ball_group = group_row["balls_band"]
            maximum = int(group_row["max_supported_wickets"])

            match_dict = test_by_group[
                (comp_group, ball_group)
            ]

            bootstrap_test = bootstrap_test_sample(
                match_dict,
                rng,
            )

            baseline_probabilities = None

            for wicket_count in range(maximum + 1):

                scenario = bootstrap_test.copy()
                scenario["wickets_lost"] = wicket_count

                scenario = add_interaction_features(scenario)

                probabilities = model.predict_proba(
                    scenario[
                        INTERACTION_NUMERIC
                        + INTERACTION_CATEGORICAL
                    ]
                )[:, 1]

                weighted_probability = np.average(
                    probabilities,
                    weights=scenario["bootstrap_weight"],
                )

                if wicket_count == 0:
                    baseline_probabilities = probabilities.copy()
                    probability_change = 0.0
                else:
                    probability_change = np.average(
                        probabilities - baseline_probabilities,
                        weights=scenario["bootstrap_weight"],
                    )

                bootstrap_results.append({
                    "iteration": iteration + 1,
                    "competition_group": comp_group,
                    "balls_band": ball_group,
                    "wickets_lost": wicket_count,
                    "mean_predicted_success_probability": (
                        weighted_probability
                    ),
                    "mean_probability_change_vs_zero": (
                        probability_change
                    ),
                })

    except Exception as error:

        failed_iterations += 1

        print(
            f"Iteration {iteration + 1} failed: {error}"
        )

    if (iteration + 1) % 25 == 0:
        print(
            f"Completed {iteration + 1}/{BOOTSTRAP_ITERATIONS} "
            f"iterations | Failed: {failed_iterations}"
        )


# ============================================================
# BOOTSTRAP COMPLETION
# ============================================================

bootstrap_df = pd.DataFrame(bootstrap_results)

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
# CONFIDENCE INTERVALS
# ============================================================

summary_rows = []

group_columns = [
    "competition_group",
    "balls_band",
    "wickets_lost",
]

for keys, group in bootstrap_df.groupby(group_columns):

    comp_group, ball_group, wicket_count = keys

    probability_values = group[
        "mean_predicted_success_probability"
    ].dropna()

    effect_values = group[
        "mean_probability_change_vs_zero"
    ].dropna()

    summary_rows.append({
        "competition_group": comp_group,
        "balls_band": ball_group,
        "wickets_lost": wicket_count,
        "successful_bootstrap_estimates": len(effect_values),

        "mean_predicted_success_probability": (
            probability_values.mean()
        ),
        "probability_ci_lower": (
            probability_values.quantile(0.025)
        ),
        "probability_ci_upper": (
            probability_values.quantile(0.975)
        ),

        "mean_probability_change_vs_zero": (
            effect_values.mean()
        ),
        "effect_ci_lower": (
            effect_values.quantile(0.025)
        ),
        "effect_ci_upper": (
            effect_values.quantile(0.975)
        ),
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUTPUT_DIR / "wicket_interaction_bootstrap_ci.csv",
    index=False,
)

bootstrap_df.to_csv(
    OUTPUT_DIR / "wicket_interaction_bootstrap_all_iterations.csv",
    index=False,
)


# ============================================================
# EFFECT STABILITY
# ============================================================

stability_rows = []

for (comp_group, ball_group), group in summary.groupby(
    ["competition_group", "balls_band"]
):

    group = group.sort_values("wickets_lost")

    non_baseline = group[
        group["wickets_lost"] > 0
    ]

    entirely_negative = (
        (non_baseline["effect_ci_upper"] < 0).all()
        if not non_baseline.empty
        else False
    )

    stability_rows.append({
        "competition_group": comp_group,
        "balls_band": ball_group,
        "wicket_scenarios_tested": len(group),
        "highest_wicket_count": group["wickets_lost"].max(),
        "all_nonzero_effect_intervals_below_zero": entirely_negative,
        "minimum_bootstrap_estimates": (
            group["successful_bootstrap_estimates"].min()
        ),
    })

stability = pd.DataFrame(stability_rows)

stability.to_csv(
    OUTPUT_DIR / "wicket_interaction_stability.csv",
    index=False,
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 95)
print("WICKET INTERACTION MODEL: BOOTSTRAP RESULTS")
print("=" * 95)

print("\n--- INTERACTION MODEL CONFIDENCE INTERVALS ---")
print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\n--- EFFECT STABILITY ---")
print(stability.to_string(index=False))

print("\n" + "=" * 95)
print(f"Outputs saved to: {OUTPUT_DIR}")
print("=" * 95)
