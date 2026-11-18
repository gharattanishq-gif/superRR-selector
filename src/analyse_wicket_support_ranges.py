
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

NUMERIC_FEATURES = [
    "wickets_lost",
    "legal_balls_remaining",
    "required_run_rate",
]

CATEGORICAL_FEATURES = ["comp_name"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# HELPER FUNCTIONS
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


def run_rate_band(value):
    if value < 6:
        return "Below 6"
    if value < 8:
        return "6-8"
    if value < 10:
        return "8-10"
    if value < 12:
        return "10-12"
    if value < 15:
        return "12-15"
    return "15+"


def build_model():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                NUMERIC_FEATURES,
            ),
            (
                "competition",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])


def get_max_supported_wickets(support_counts):
    """
    Return the highest consecutive wicket count starting at zero
    for which every wicket level meets the support threshold.

    Example:
        0-5 supported, 6 unsupported -> returns 5.
        0 unsupported -> returns -1.
    """

    maximum = -1

    for wicket_count in range(10):
        count = support_counts.get(wicket_count, 0)

        if count >= MIN_SUPPORT_MATCHES:
            maximum = wicket_count
        else:
            break

    return maximum


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

balls = pd.read_csv(BALLS_FILE, low_memory=False)
matches = pd.read_csv(MATCHES_FILE, low_memory=False)
competitions = pd.read_csv(COMPETITIONS_FILE, low_memory=False)

numeric_columns = [
    "runs",
    "extras1_runs",
    "extras_2_runs",
    "innings",
    "overid",
    "over_ball_num",
    "is_wicket",
]

for col in numeric_columns:
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

# Provisional treatment: wides and oversteps are illegal balls.
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

# One pre-delivery checkpoint every six legal balls.
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
        "required_run_rate",
        "chase_success",
    ]
].copy()

model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan,
)

model_data = model_data.dropna(
    subset=FEATURES + [
        "match_id",
        "match_date",
        "chase_success",
    ]
)

model_data["chase_success"] = model_data[
    "chase_success"
].astype(int)

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
# TRAIN LOGISTIC REGRESSION
# ============================================================

print("\nTraining original logistic regression...")

train["match_weight"] = (
    1 / train.groupby("match_id")["match_id"].transform("size")
)

model = build_model()

model.fit(
    train[FEATURES],
    train["chase_success"],
    model__sample_weight=train["match_weight"],
)


# ============================================================
# DEFINE SUPPORT GROUPS
# ============================================================

for dataset in [train, test]:
    dataset["competition_group"] = (
        dataset["comp_name"].apply(competition_group)
    )

    dataset["balls_band"] = (
        dataset["legal_balls_remaining"].apply(balls_band)
    )

    dataset["run_rate_band"] = (
        dataset["required_run_rate"].apply(run_rate_band)
    )


# ============================================================
# HISTORICAL SUPPORT COUNTS
# ============================================================

print("\nCalculating historical support...")

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

print("\n--- HISTORICAL SUPPORT COUNTS ---")
print(support.to_string(index=False))


# ============================================================
# FIND MAXIMUM CONSECUTIVE WICKET RANGE
# ============================================================

print("\nFinding supported wicket ranges...")

test_groups = (
    test[
        ["competition_group", "balls_band"]
    ]
    .drop_duplicates()
    .sort_values(
        ["competition_group", "balls_band"]
    )
)

range_rows = []

support_lookup = {}

for _, row in support.iterrows():
    key = (
        row["competition_group"],
        row["balls_band"],
        int(row["wickets_lost"]),
    )
    support_lookup[key] = int(row["support_matches"])


for _, row in test_groups.iterrows():

    comp_group = row["competition_group"]
    ball_group = row["balls_band"]

    counts = {
        wicket_count: support_lookup.get(
            (comp_group, ball_group, wicket_count),
            0,
        )
        for wicket_count in range(10)
    }

    maximum = get_max_supported_wickets(counts)

    test_subset = test[
        (test["competition_group"] == comp_group)
        & (test["balls_band"] == ball_group)
    ]

    range_rows.append({
        "competition_group": comp_group,
        "balls_band": ball_group,
        "max_supported_wickets": maximum,
        "supported_range": (
            f"0-{maximum}" if maximum >= 0 else "No supported baseline"
        ),
        "test_states": len(test_subset),
        "test_matches": test_subset["match_id"].nunique(),
        "historical_support_at_zero": counts[0],
        "historical_support_at_max": (
            counts[maximum] if maximum >= 0 else 0
        ),
        "historical_support_at_next": (
            counts[maximum + 1] if maximum < 9 else np.nan
        ),
    })

support_ranges = pd.DataFrame(range_rows)

print("\n--- SUPPORTED WICKET RANGES ---")
print(support_ranges.to_string(index=False))


# ============================================================
# CALCULATE EFFECTS WITHIN EACH SUPPORTED RANGE
# ============================================================

print("\nCalculating marginal effects within supported ranges...")

scenario_frames = []
effect_rows = []
coverage_rows = []

for _, range_row in support_ranges.iterrows():

    comp_group = range_row["competition_group"]
    ball_group = range_row["balls_band"]
    maximum = int(range_row["max_supported_wickets"])

    if maximum < 0:
        continue

    group_test = test[
        (test["competition_group"] == comp_group)
        & (test["balls_band"] == ball_group)
    ].copy()

    if group_test.empty:
        continue

    # Identical test situations are used for every wicket scenario
    # within this competition and balls-remaining phase.

    # Equal total weight per match within this comparison group.
    group_test["match_weight"] = (
        1
        / group_test.groupby("match_id")["match_id"].transform("size")
    )

    scenario_group_frames = []

    for wicket_count in range(maximum + 1):

        scenario = group_test[
            [
                "match_id",
                "match_date",
                "comp_name",
                "season",
                "competition_group",
                "balls_band",
                "run_rate_band",
                "legal_balls_remaining",
                "required_run_rate",
                "match_weight",
            ]
        ].copy()

        scenario["wickets_lost"] = wicket_count

        scenario["predicted_chase_success"] = (
            model.predict_proba(
                scenario[FEATURES]
            )[:, 1]
        )

        scenario_group_frames.append(scenario)

    group_scenarios = pd.concat(
        scenario_group_frames,
        ignore_index=True,
    )

    baseline = group_scenarios[
        group_scenarios["wickets_lost"] == 0
    ][
        [
            "match_id",
            "legal_balls_remaining",
            "required_run_rate",
            "comp_name",
            "predicted_chase_success",
        ]
    ].rename(
        columns={
            "predicted_chase_success": "zero_wicket_probability"
        }
    )

    group_scenarios = group_scenarios.merge(
        baseline,
        on=[
            "match_id",
            "legal_balls_remaining",
            "required_run_rate",
            "comp_name",
        ],
        how="inner",
        validate="many_to_one",
    )

    group_scenarios["probability_change_vs_zero"] = (
        group_scenarios["predicted_chase_success"]
        - group_scenarios["zero_wicket_probability"]
    )

    # Verify identical population within this group.
    population_check = (
        group_scenarios.groupby("wickets_lost")
        .agg(
            matches=("match_id", "nunique"),
            states=("match_id", "size"),
        )
    )

    if (
        population_check["matches"].nunique() != 1
        or population_check["states"].nunique() != 1
    ):
        raise ValueError(
            f"Population mismatch in {comp_group}, {ball_group}."
        )

    for wicket_count, scenario in group_scenarios.groupby(
        "wickets_lost"
    ):

        effect_rows.append({
            "competition_group": comp_group,
            "balls_band": ball_group,
            "max_supported_wickets": maximum,
            "wickets_lost": wicket_count,
            "supported_matches": scenario["match_id"].nunique(),
            "supported_states": len(scenario),
            "mean_predicted_success_probability": np.average(
                scenario["predicted_chase_success"],
                weights=scenario["match_weight"],
            ),
            "mean_probability_change_vs_zero": np.average(
                scenario["probability_change_vs_zero"],
                weights=scenario["match_weight"],
            ),
            "median_probability_change_vs_zero": (
                scenario["probability_change_vs_zero"].median()
            ),
        })

    coverage_rows.append({
        "competition_group": comp_group,
        "balls_band": ball_group,
        "max_supported_wickets": maximum,
        "supported_range": f"0-{maximum}",
        "test_matches": group_test["match_id"].nunique(),
        "test_states": len(group_test),
        "scenario_count": maximum + 1,
    })

    scenario_frames.append(group_scenarios)


# ============================================================
# COMBINE RESULTS
# ============================================================

effects = pd.DataFrame(effect_rows)
coverage = pd.DataFrame(coverage_rows)

if not scenario_frames:
    raise ValueError(
        "No supported comparison groups were found."
    )

all_scenarios = pd.concat(
    scenario_frames,
    ignore_index=True,
)


# ============================================================
# RUN-RATE-SPECIFIC EFFECTS
# ============================================================

run_rate_rows = []

for keys, group in all_scenarios.groupby(
    [
        "competition_group",
        "balls_band",
        "run_rate_band",
        "wickets_lost",
    ]
):

    comp_group, ball_group, rr_group, wicket_count = keys

    weights = group["match_weight"]

    run_rate_rows.append({
        "competition_group": comp_group,
        "balls_band": ball_group,
        "run_rate_band": rr_group,
        "wickets_lost": wicket_count,
        "supported_matches": group["match_id"].nunique(),
        "supported_states": len(group),
        "mean_predicted_success_probability": np.average(
            group["predicted_chase_success"],
            weights=weights,
        ),
        "mean_probability_change_vs_zero": np.average(
            group["probability_change_vs_zero"],
            weights=weights,
        ),
    })

run_rate_effects = pd.DataFrame(run_rate_rows)


# ============================================================
# SAVE OUTPUTS
# ============================================================

support.to_csv(
    OUTPUT_DIR / "wicket_support_range_historical_counts.csv",
    index=False,
)

support_ranges.to_csv(
    OUTPUT_DIR / "wicket_support_ranges_by_phase.csv",
    index=False,
)

effects.to_csv(
    OUTPUT_DIR / "wicket_marginal_effects_by_supported_range.csv",
    index=False,
)

coverage.to_csv(
    OUTPUT_DIR / "wicket_support_range_coverage.csv",
    index=False,
)

run_rate_effects.to_csv(
    OUTPUT_DIR / "wicket_marginal_effects_by_run_rate.csv",
    index=False,
)

all_scenarios.to_csv(
    OUTPUT_DIR / "wicket_support_range_scenarios.csv",
    index=False,
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 85)
print("EXPANDED WICKET SUPPORT ANALYSIS")
print("=" * 85)

print("\n--- SUPPORTED WICKET RANGES ---")
print(
    support_ranges.to_string(
        index=False,
    )
)

print("\n--- COVERAGE ---")
if not coverage.empty:
    print(coverage.to_string(index=False))

print("\n--- MARGINAL EFFECTS BY SUPPORTED RANGE ---")
if not effects.empty:
    print(
        effects.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

print("\n--- MARGINAL EFFECTS BY REQUIRED RUN RATE ---")
if not run_rate_effects.empty:
    print(
        run_rate_effects.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

print("\n" + "=" * 85)
print(f"Outputs saved to: {OUTPUT_DIR}")
print("=" * 85)
