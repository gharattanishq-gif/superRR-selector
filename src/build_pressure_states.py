
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "Data Pack"
PROCESSED_DIR = ROOT / "outputs" / "processed"
CHASE_DIR = ROOT / "outputs" / "analysis" / "chase_adjustment"
OUTPUT_DIR = ROOT / "outputs" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BALLS_FILE = DATA_DIR / "ball_by_ball.csv"
MATCHES_FILE = DATA_DIR / "matches.csv"
COMPETITIONS_FILE = DATA_DIR / "competitions.csv"

OUTPUT_FILE = OUTPUT_DIR / "pressure_states.csv"
COVERAGE_FILE = OUTPUT_DIR / "pressure_prediction_coverage.csv"
VALIDATION_FILE = OUTPUT_DIR / "pressure_state_validation.csv"

ANALYSIS_START_DATE = pd.Timestamp("2024-01-01")
N_PREDICTION_BLOCKS = 6

MIN_FIRST_INNINGS_TRAIN_MATCHES = 250
MIN_CHASE_TRAIN_MATCHES = 200
MIN_CHASE_SUPPORT_MATCHES = 5

CHECKPOINT_INTERVAL = 6
RANDOM_SEED = 42

# Previously established full-overs competitive medians.
PAR_SCORES = {
    "IPL": 183.0,
    "T20I": 169.0,
}


# ============================================================
# 2. MODEL FEATURES
# ============================================================

FIRST_NUMERIC = [
    "current_score",
    "wickets_lost",
    "legal_balls_remaining",
]

FIRST_CATEGORICAL = [
    "comp_name",
    "season",
    "venue_id",
]

CHASE_NUMERIC = [
    "wickets_lost",
    "legal_balls_remaining",
    "required_run_rate",
    "run_rate_gap",
    "wickets_x_phase_1_12",
    "wickets_x_phase_13_30",
    "wickets_x_phase_61_90",
    "wickets_x_phase_91_120",
    "wickets_x_t20i",
]

CHASE_CATEGORICAL = [
    "comp_name",
    "balls_band",
    "competition_group",
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

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


def equal_match_weights(data):
    counts = data.groupby("match_id")["match_id"].transform("size")
    return 1.0 / counts


def make_expanding_folds(match_context):
    """
    Use all matches before 1 January 2024 as historical training data.
    Generate out-of-sample predictions only for matches dated 2024 onward.

    Later folds expand the training window with earlier 2024+ matches.
    Matches on the same calendar date always remain in one partition.
    """

    dates = (
        match_context[["match_id", "match_date"]]
        .drop_duplicates()
        .sort_values(["match_date", "match_id"])
    )

    dates = dates.dropna(subset=["match_date"]).copy()

    historical_dates = dates[
        dates["match_date"] < ANALYSIS_START_DATE
    ]

    prediction_dates = dates[
        dates["match_date"] >= ANALYSIS_START_DATE
    ]

    unique_prediction_dates = np.array(
        sorted(prediction_dates["match_date"].unique())
    )

    if len(unique_prediction_dates) == 0:
        raise ValueError(
            "No matches found on or after 2024-01-01."
        )

    date_blocks = np.array_split(
        unique_prediction_dates,
        N_PREDICTION_BLOCKS,
    )

    folds = []

    for block_number, block_dates in enumerate(date_blocks, start=1):

        if len(block_dates) == 0:
            continue

        first_test_date = pd.Timestamp(block_dates.min())

        # Expanding window: all historical data strictly before this block.
        train_ids = set(
            dates.loc[
                dates["match_date"] < first_test_date,
                "match_id",
            ]
        )

        test_ids = set(
            prediction_dates.loc[
                prediction_dates["match_date"].isin(block_dates),
                "match_id",
            ]
        )

        if not train_ids or not test_ids:
            continue

        assert train_ids.isdisjoint(test_ids)

        # No pre-2024 match can enter the prediction set.
        assert all(
            dates.loc[dates["match_id"].isin(test_ids), "match_date"]
            >= ANALYSIS_START_DATE
        )

        folds.append({
            "block": block_number,
            "train_ids": train_ids,
            "test_ids": test_ids,
            "train_matches": len(train_ids),
            "test_matches": len(test_ids),
            "historical_training_matches": int(
                dates.loc[
                    dates["match_id"].isin(train_ids)
                    & (dates["match_date"] < ANALYSIS_START_DATE),
                    "match_id",
                ].nunique()
            ),
            "2024_plus_training_matches": int(
                dates.loc[
                    dates["match_id"].isin(train_ids)
                    & (dates["match_date"] >= ANALYSIS_START_DATE),
                    "match_id",
                ].nunique()
            ),
            "train_through": dates.loc[
                dates["match_id"].isin(train_ids),
                "match_date",
            ].max(),
            "test_from": first_test_date,
            "test_to": pd.Timestamp(block_dates.max()),
        })

    if not folds:
        raise ValueError(
            "Could not create expanding prediction folds."
        )

    return folds


def add_chase_interactions(data):

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


def build_chase_model():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            CHASE_NUMERIC,
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CHASE_CATEGORICAL,
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


def clean_extra_type(series):
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\nLoading source datasets...")
print("Final analysis window: 2024 onward")
print("Historical training data: all available matches before each prediction block")

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
    balls[col] = numeric(balls[col]).fillna(0)

matches["match_date"] = pd.to_datetime(
    matches["match_date"],
    errors="coerce",
)

matches["win_team_id_num"] = numeric(
    matches["win_team_id"]
)

context = matches.merge(
    competitions[
        ["comp_id", "comp_name", "match_format", "season"]
    ],
    on="comp_id",
    how="left",
)

competition_name = (
    context["comp_name"].fillna("").str.lower()
)

match_format = (
    context["match_format"].fillna("").str.lower()
)

context["is_primary"] = (
    competition_name.str.contains("ipl")
    | match_format.eq("t20i")
)

context = context[
    context["is_primary"]
].dropna(
    subset=["match_id", "match_date", "comp_name"]
).copy()

context["season"] = context["season"].astype(str)

context["first_max_balls"] = (
    numeric(context["first_inn_revised_overs"])
    .fillna(120)
    .astype(int)
)

context["second_max_balls"] = (
    numeric(context["second_inn_revised_overs"])
    .fillna(120)
    .astype(int)
)

context["competition_group"] = (
    context["comp_name"].apply(competition_group)
)

print("Primary competition matches:", len(context))
print(
    "Competition groups:",
    context["competition_group"].value_counts().to_dict(),
)


# ============================================================
# 5. DELIVERY PROCESSING
# ============================================================

balls = balls[
    balls["match_id"].isin(set(context["match_id"]))
].copy()

balls["extras_type_clean"] = clean_extra_type(
    balls["extras_type"]
)

# Preserve the provisional delivery treatment from earlier work.
balls["legal_ball"] = ~balls[
    "extras_type_clean"
].isin(["wide", "overstep"])

balls["unresolved_extra"] = balls[
    "extras_type_clean"
].isin(["other", "penalty"])

balls["delivery_runs"] = (
    balls["runs"]
    + balls["extras1_runs"]
    + balls["extras_2_runs"]
)

balls["is_wicket"] = balls["is_wicket"].astype(int)

balls = balls.merge(
    context[
        [
            "match_id",
            "match_date",
            "comp_name",
            "season",
            "venue_id",
            "first_max_balls",
            "second_max_balls",
            "win_team_id_num",
            "competition_group",
        ]
    ],
    on="match_id",
    how="inner",
)

balls = balls.sort_values(
    ["match_id", "innings", "overid", "over_ball_num"]
).reset_index(drop=True)

# Retain a stable source-row identifier.
balls["source_row_id"] = np.arange(len(balls))


# ============================================================
# 6. FIRST-INNINGS DELIVERY STATES
# ============================================================

print("\nBuilding first-innings states...")

first = balls[
    balls["innings"] == 1
].copy()

first_group = first.groupby(
    "match_id",
    sort=False,
)

first["innings_has_unresolved"] = (
    first_group["unresolved_extra"].transform("max")
)

first["innings_legal_balls"] = (
    first_group["legal_ball"].transform("sum")
)

first["valid_innings"] = (
    ~first["innings_has_unresolved"]
    & (
        first["innings_legal_balls"]
        <= first["first_max_balls"]
    )
)

first = first[
    first["valid_innings"]
].copy()

first_group = first.groupby(
    "match_id",
    sort=False,
)

first["current_score"] = (
    first_group["delivery_runs"].cumsum()
    - first["delivery_runs"]
)

first["wickets_lost"] = (
    first_group["is_wicket"].cumsum()
    - first["is_wicket"]
)

first["legal_balls_bowled"] = (
    first_group["legal_ball"].cumsum().astype(int)
    - first["legal_ball"].astype(int)
)

first["legal_balls_remaining"] = (
    first["first_max_balls"]
    - first["legal_balls_bowled"]
)

first["final_score"] = (
    first_group["delivery_runs"].transform("sum")
)

first["is_full_overs"] = (
    first["first_max_balls"] == 120
)

first = first[
    (first["legal_balls_remaining"] > 0)
    & (first["wickets_lost"] <= 10)
].copy()

first = first.replace(
    [np.inf, -np.inf],
    np.nan,
)

first = first.dropna(
    subset=[
        "match_id",
        "match_date",
        "current_score",
        "wickets_lost",
        "legal_balls_remaining",
        "final_score",
    ]
)

first["venue_id"] = first["venue_id"].astype(str)
first["season"] = first["season"].astype(str)
first["comp_name"] = first["comp_name"].astype(str)

first_match_context = (
    first[["match_id", "match_date"]]
    .drop_duplicates()
)

first_folds = make_expanding_folds(
    first_match_context
)

print("First-innings state rows:", len(first))
print(
    "First-innings matches:",
    first["match_id"].nunique(),
)
print("First-innings prediction blocks:", len(first_folds))


# ============================================================
# 7. FIRST-INNINGS EXPANDING-WINDOW PREDICTIONS
# ============================================================

print("\nGenerating first-innings predictions...")

first_predictions = []
first_coverage = []

for fold in first_folds:

    train = first[
        first["match_id"].isin(fold["train_ids"])
    ].copy()

    test = first[
        first["match_id"].isin(fold["test_ids"])
    ].copy()

    train_match_count = train["match_id"].nunique()

    if train_match_count < MIN_FIRST_INNINGS_TRAIN_MATCHES:
        print(
            f"Skipping first-innings block {fold['block']}: "
            f"only {train_match_count} training matches."
        )
        continue

    model = Pipeline([
        (
            "preprocessor",
            ColumnTransformer([
                (
                    "numeric",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    FIRST_NUMERIC,
                ),
                (
                    "categorical",
                    Pipeline([
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]),
                    FIRST_CATEGORICAL,
                ),
            ]),
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=5,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        ),
    ])

    model.fit(
        train[FIRST_NUMERIC + FIRST_CATEGORICAL],
        train["final_score"],
    )

    test["predicted_final_score"] = model.predict(
        test[FIRST_NUMERIC + FIRST_CATEGORICAL]
    )

    test["prediction_block"] = fold["block"]
    test["training_matches"] = train_match_count
    test["prediction_training_through"] = fold["train_through"]

    first_predictions.append(test)

    first_coverage.append({
        "model": "First innings Random Forest",
        "block": fold["block"],
        "training_matches": train_match_count,
        "prediction_matches": test["match_id"].nunique(),
        "training_through": fold["train_through"],
        "prediction_from": fold["test_from"],
        "prediction_to": fold["test_to"],
        "prediction_rows": len(test),
    })

if not first_predictions:
    raise ValueError(
        "No first-innings predictions generated. "
        "Check training-match requirements."
    )

first_pred = pd.concat(
    first_predictions,
    ignore_index=True,
)

first_pred["par_score"] = (
    first_pred["competition_group"].map(PAR_SCORES)
)

first_pred["first_innings_batting_pressure"] = (
    first_pred["par_score"]
    - first_pred["predicted_final_score"]
)

first_pred["first_innings_bowling_pressure"] = (
    first_pred["predicted_final_score"]
    - first_pred["par_score"]
)

# Only full-overs matches have an established par benchmark.
first_pred["pressure_eligible"] = (
    first_pred["is_full_overs"]
    & first_pred["par_score"].notna()
)

first_pred.loc[
    ~first_pred["pressure_eligible"],
    [
        "first_innings_batting_pressure",
        "first_innings_bowling_pressure",
    ],
] = np.nan

print(
    "First-innings predicted matches:",
    first_pred["match_id"].nunique(),
)


# ============================================================
# 8. SECOND-INNINGS MATCH OUTCOMES
# ============================================================

print("\nBuilding second-innings states...")

innings_balls = balls[
    balls["innings"].isin([1, 2])
].copy()

first_outcomes = (
    innings_balls[
        innings_balls["innings"] == 1
    ]
    .groupby("match_id", as_index=False)
    .agg(
        first_innings_score=("delivery_runs", "sum"),
        first_innings_unresolved=("unresolved_extra", "any"),
    )
)

second_metadata = (
    innings_balls[
        innings_balls["innings"] == 2
    ]
    .groupby("match_id", as_index=False)
    .agg(
        chasing_team_id=("batting_team_id", "first"),
        second_innings_unresolved=("unresolved_extra", "any"),
        second_batting_teams=("batting_team_id", "nunique"),
    )
)

chase_matches = (
    context.merge(
        first_outcomes,
        on="match_id",
        how="inner",
    )
    .merge(
        second_metadata,
        on="match_id",
        how="inner",
    )
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

chase_matches["target"] = numeric(
    chase_matches["revised_target"]
)

chase_matches["target"] = (
    chase_matches["target"].fillna(
        chase_matches["first_innings_score"] + 1
    )
)

chase_matches = chase_matches[
    chase_matches["target"] > 0
].copy()

print("Usable chase matches:", len(chase_matches))
print(
    "Successful chases:",
    int(chase_matches["chase_success"].sum()),
)


# ============================================================
# 9. SECOND-INNINGS DELIVERY STATES
# ============================================================

second = innings_balls[
    innings_balls["innings"] == 2
].merge(
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

second = second.sort_values(
    ["match_id", "overid", "over_ball_num"]
).copy()

# Exclude unresolved deliveries from state construction.
second = second[
    ~second["unresolved_extra"]
].copy()

second_group = second.groupby(
    "match_id",
    sort=False,
)

second["current_score"] = (
    second_group["delivery_runs"].cumsum()
    - second["delivery_runs"]
)

second["wickets_lost"] = (
    second_group["is_wicket"].cumsum()
    - second["is_wicket"]
)

second["legal_balls_bowled"] = (
    second_group["legal_ball"].cumsum().astype(int)
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

# Current run rate is calculated from the state before the current delivery.
# At the start of an innings, no balls have been faced, so CRR is undefined.
second["current_run_rate"] = np.where(
    second["legal_balls_bowled"] > 0,
    second["current_score"] * 6
    / second["legal_balls_bowled"].replace(0, np.nan),
    np.nan,
)

# Model C's shared RRR-CRR gap feature.
second["run_rate_gap"] = (
    second["required_run_rate"] - second["current_run_rate"]
)

second["balls_band"] = (
    second["legal_balls_remaining"].apply(balls_band)
)

second = add_chase_interactions(second)

second = second.replace(
    [np.inf, -np.inf],
    np.nan,
)

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

second["chase_success"] = second[
    "chase_success"
].astype(int)

second_match_context = (
    second[["match_id", "match_date"]]
    .drop_duplicates()
)

chase_folds = make_expanding_folds(
    second_match_context
)

print("Second-innings state rows:", len(second))
print(
    "Second-innings matches:",
    second["match_id"].nunique(),
)
print("Chase prediction blocks:", len(chase_folds))


# ============================================================
# 10. CHASE MODEL EXPANDING-WINDOW PREDICTIONS
# ============================================================

print("\nGenerating chase probabilities...")

chase_predictions = []
chase_coverage = []

for fold in chase_folds:

    train_all = second[
        second["match_id"].isin(fold["train_ids"])
    ].copy()

    test = second[
        second["match_id"].isin(fold["test_ids"])
    ].copy()

    train_match_count = train_all["match_id"].nunique()

    if train_match_count < MIN_CHASE_TRAIN_MATCHES:
        print(
            f"Skipping chase block {fold['block']}: "
            f"only {train_match_count} training matches."
        )
        continue

    # The original model was trained on six-ball checkpoints.
    train = train_all[
        train_all["legal_balls_remaining"]
        % CHECKPOINT_INTERVAL == 0
    ].copy()

    if train["chase_success"].nunique() < 2:
        continue

    train["match_weight"] = equal_match_weights(train)

    model = build_chase_model()

    model.fit(
        train[CHASE_NUMERIC + CHASE_CATEGORICAL],
        train["chase_success"],
        model__sample_weight=train["match_weight"],
    )

    # Historical support at competition / phase / wicket level.
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

    # Required-run-rate range within competition and phase.
    rrr_ranges = (
        train.groupby(
            ["competition_group", "balls_band"]
        )["required_run_rate"]
        .quantile([0.01, 0.99])
        .unstack()
        .reset_index()
        .rename(columns={0.01: "rrr_p01", 0.99: "rrr_p99"})
    )

    test = test.merge(
        rrr_ranges,
        on=["competition_group", "balls_band"],
        how="left",
    )

    test["support_matches"] = [
        support_lookup.get(
            (
                row.competition_group,
                row.balls_band,
                int(row.wickets_lost),
            ),
            0,
        )
        for row in test.itertuples()
    ]

    test["chase_prediction_supported"] = (
        (test["support_matches"] >= MIN_CHASE_SUPPORT_MATCHES)
        & (test["required_run_rate"] >= test["rrr_p01"])
        & (test["required_run_rate"] <= test["rrr_p99"])
    )

    # Predict only historically supported states.
    supported_mask = test["chase_prediction_supported"]

    test["predicted_chase_success"] = np.nan

    if supported_mask.any():

        probabilities = model.predict_proba(
            test.loc[
                supported_mask,
                CHASE_NUMERIC + CHASE_CATEGORICAL,
            ]
        )[:, 1]

        test.loc[
            supported_mask,
            "predicted_chase_success",
        ] = probabilities

    test["second_innings_batting_pressure"] = (
        1 - test["predicted_chase_success"]
    )

    test["second_innings_bowling_pressure"] = (
        test["predicted_chase_success"]
    )

    test["prediction_block"] = fold["block"]
    test["training_matches"] = train_match_count
    test["prediction_training_through"] = fold["train_through"]

    chase_predictions.append(test)

    chase_coverage.append({
        "model": "Interaction logistic regression + shared RRR-CRR gap (Model C)",
        "block": fold["block"],
        "training_matches": train_match_count,
        "prediction_matches": test["match_id"].nunique(),
        "training_through": fold["train_through"],
        "prediction_from": fold["test_from"],
        "prediction_to": fold["test_to"],
        "prediction_rows": len(test),
        "supported_prediction_rows": int(supported_mask.sum()),
        "unsupported_prediction_rows": int((~supported_mask).sum()),
    })

if not chase_predictions:
    raise ValueError(
        "No chase predictions generated. "
        "Check training-match requirements."
    )

chase_pred = pd.concat(
    chase_predictions,
    ignore_index=True,
)

print(
    "Chase predicted matches:",
    chase_pred["match_id"].nunique(),
)


# ============================================================
# 11. STANDARDISE OUTPUT COLUMNS
# ============================================================

COMMON_COLUMNS = [
    "source_row_id",
    "match_id",
    "match_date",
    "comp_name",
    "season",
    "competition_group",
    "innings",
    "overid",
    "over_ball_num",
    "innings_ball_num",
    "striker_id",
    "striker_name",
    "non_striker_id",
    "non_striker_name",
    "bowler_id",
    "bowler_name",
    "batting_team_id",
    "bowling_team_id",
    "runs",
    "is_wicket",
    "legal_ball",
    "unresolved_extra",
]


first_output = first_pred.copy()

first_output["innings"] = 1
first_output["pressure_method"] = "First innings expected score vs par"

first_output["batting_pressure"] = (
    first_output["first_innings_batting_pressure"]
)

first_output["bowling_pressure"] = (
    first_output["first_innings_bowling_pressure"]
)

first_output["pressure_eligible"] = (
    first_output["pressure_eligible"]
)

first_output["predicted_chase_success"] = np.nan
first_output["chase_prediction_supported"] = False

first_output["expected_final_score"] = (
    first_output["predicted_final_score"]
)

first_output["par_score"] = first_output["par_score"]


second_output = chase_pred.copy()

second_output["innings"] = 2
second_output["pressure_method"] = "Model C raw chase-success probability"

second_output["batting_pressure"] = (
    second_output["second_innings_batting_pressure"]
)

second_output["bowling_pressure"] = (
    second_output["second_innings_bowling_pressure"]
)

second_output["pressure_eligible"] = (
    second_output["chase_prediction_supported"]
)

second_output["expected_final_score"] = np.nan
second_output["par_score"] = np.nan


# Ensure all expected columns exist.
for frame in [first_output, second_output]:
    for col in COMMON_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan

    for col in [
        "batting_pressure",
        "bowling_pressure",
        "pressure_eligible",
        "pressure_method",
        "expected_final_score",
        "par_score",
        "predicted_chase_success",
        "chase_prediction_supported",
        "prediction_block",
        "training_matches",
        "prediction_training_through",
        "support_matches",
    ]:
        if col not in frame.columns:
            frame[col] = np.nan


OUTPUT_COLUMNS = COMMON_COLUMNS + [
    "current_score",
    "wickets_lost",
    "legal_balls_remaining",
    "runs_required",
    "required_run_rate",
    "current_run_rate",
    "run_rate_gap",
    "expected_final_score",
    "par_score",
    "predicted_chase_success",
    "batting_pressure",
    "bowling_pressure",
    "pressure_method",
    "pressure_eligible",
    "chase_prediction_supported",
    "support_matches",
    "prediction_block",
    "training_matches",
    "prediction_training_through",
]


first_output = first_output.reindex(
    columns=OUTPUT_COLUMNS
)

second_output = second_output.reindex(
    columns=OUTPUT_COLUMNS
)

pressure_states = pd.concat(
    [first_output, second_output],
    ignore_index=True,
)

pressure_states = pressure_states.sort_values(
    ["match_date", "match_id", "innings", "overid", "over_ball_num"]
).reset_index(drop=True)


# ============================================================
# 12. VALIDATION
# ============================================================

print("\nValidating pressure states...")

validation_rows = []

for innings_number, group in pressure_states.groupby("innings"):

    eligible = group[
        group["pressure_eligible"] == True
    ]

    validation_rows.append({
        "innings": innings_number,
        "total_prediction_rows": len(group),
        "matches": group["match_id"].nunique(),
        "eligible_pressure_rows": len(eligible),
        "eligible_matches": eligible["match_id"].nunique(),
        "missing_batting_pressure": int(
            eligible["batting_pressure"].isna().sum()
        ),
        "missing_bowling_pressure": int(
            eligible["bowling_pressure"].isna().sum()
        ),
        "negative_batting_pressure": int(
            (eligible["batting_pressure"] < 0).sum()
        ),
        "negative_bowling_pressure": int(
            (eligible["bowling_pressure"] < 0).sum()
        ),
    })

validation = pd.DataFrame(validation_rows)

# Check for duplicate source delivery IDs.
duplicate_rows = int(
    pressure_states.duplicated(
        ["source_row_id"]
    ).sum()
)

print(validation.to_string(index=False))
print("Duplicate source rows:", duplicate_rows)

if duplicate_rows:
    raise ValueError(
        "Duplicate source delivery rows detected."
    )

# First-innings pressure is only eligible for full overs.
invalid_first = pressure_states[
    (pressure_states["innings"] == 1)
    & pressure_states["pressure_eligible"].eq(True)
    & pressure_states["par_score"].isna()
]

if len(invalid_first):
    raise ValueError(
        "First-innings eligible rows contain missing par scores."
    )

# Second-innings pressure must have supported probabilities.
invalid_second = pressure_states[
    (pressure_states["innings"] == 2)
    & pressure_states["pressure_eligible"].eq(True)
    & pressure_states["predicted_chase_success"].isna()
]

if len(invalid_second):
    raise ValueError(
        "Second-innings eligible rows contain missing probabilities."
    )


# ============================================================
# 13. SAVE OUTPUTS
# ============================================================

coverage = pd.DataFrame(
    first_coverage + chase_coverage
)

pressure_states.to_csv(
    OUTPUT_FILE,
    index=False,
)

coverage.to_csv(
    COVERAGE_FILE,
    index=False,
)

validation.to_csv(
    VALIDATION_FILE,
    index=False,
)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("PRESSURE STATE DATASET COMPLETE")
print("=" * 75)

print("Total pressure-state rows:", len(pressure_states))
print("Total matches:", pressure_states["match_id"].nunique())

print("\nRows by innings:")
print(
    pressure_states.groupby("innings").size().to_string()
)

print("\nEligible rows by innings:")
print(
    pressure_states[
        pressure_states["pressure_eligible"] == True
    ].groupby("innings").size().to_string()
)

print("\nEligible matches by innings:")
print(
    pressure_states[
        pressure_states["pressure_eligible"] == True
    ].groupby("innings")["match_id"].nunique().to_string()
)

print("\nPrediction coverage:")
print(coverage.to_string(index=False))

print("\nFiles saved:")
print(OUTPUT_FILE)
print(COVERAGE_FILE)
print(VALIDATION_FILE)

print("\n" + "=" * 75)
