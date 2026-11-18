
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

# Sample one pre-delivery checkpoint every six legal balls.
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

# Equal total training weight per match.
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
# HISTORICAL SUPPORT
# ============================================================

print("\nChecking historical support...")

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

# Count distinct training matches supporting each wicket scenario.
support = (
    train.groupby(
        [
            "competition_group",
            "wickets_lost",
            "balls_band",
        ]
    )["match_id"]
    .nunique()
    .reset_index(name="support_matches")
)

print("\n--- HISTORICAL SUPPORT ---")
print(support.to_string(index=False))

support_lookup = support.set_index(
    [
        "competition_group",
        "wickets_lost",
        "balls_band",
    ]
)["support_matches"]


# ============================================================
# COMMON SUPPORT ACROSS ALL TEN SCENARIOS
# ============================================================

print("\nIdentifying common-support test situations...")

# For every test situation, find the minimum historical support
# across all hypothetical wicket counts from 0 to 9.

common_support = test[
    [
        "match_id",
        "match_date",
        "comp_name",
        "season",
        "competition_group",
        "legal_balls_remaining",
        "required_run_rate",
        "balls_band",
        "run_rate_band",
    ]
].copy()

for wicket_count in range(10):

    keys = pd.MultiIndex.from_arrays([
        common_support["competition_group"],
        np.full(len(common_support), wicket_count),
        common_support["balls_band"],
    ])

    common_support[f"support_w{wicket_count}"] = (
        support_lookup.reindex(keys)
        .fillna(0)
        .to_numpy()
    )

support_columns = [
    f"support_w{i}" for i in range(10)
]

common_support["minimum_scenario_support"] = (
    common_support[support_columns].min(axis=1)
)

common_support["common_support_eligible"] = (
    common_support["minimum_scenario_support"]
    >= MIN_SUPPORT_MATCHES
)

support_diagnostics = (
    common_support.groupby(
        [
            "competition_group",
            "balls_band",
        ],
        dropna=False,
    )
    .agg(
        test_states=("match_id", "size"),
        eligible_states=("common_support_eligible", "sum"),
        eligible_matches=(
            "match_id",
            lambda x: x[
                common_support.loc[x.index, "common_support_eligible"]
            ].nunique(),
        ),
        minimum_historical_support=(
            "minimum_scenario_support",
            "max",
        ),
    )
    .reset_index()
)

print("\n--- COMMON SUPPORT DIAGNOSTICS ---")
print(support_diagnostics.to_string(index=False))

eligible_keys = common_support.loc[
    common_support["common_support_eligible"],
    [
        "match_id",
        "legal_balls_remaining",
    ],
].drop_duplicates()

common_test = test.merge(
    eligible_keys,
    on=[
        "match_id",
        "legal_balls_remaining",
    ],
    how="inner",
)

if common_test.empty:
    raise ValueError(
        "No test situations have sufficient common support "
        "across all ten wicket scenarios. Do not lower the "
        "threshold automatically. Review the support diagnostics."
    )

print("\n--- COMMON TEST POPULATION ---")
print(f"Eligible states: {len(common_test):,}")
print(f"Eligible matches: {common_test['match_id'].nunique()}")
print(
    "Share of original test states:",
    f"{len(common_test) / len(test):.1%}",
)


# ============================================================
# REBUILD MATCH WEIGHTS AFTER COMMON-SUPPORT FILTERING
# ============================================================

# Every eligible match contributes total weight = 1.
common_test["match_weight"] = (
    1
    / common_test.groupby("match_id")["match_id"].transform("size")
)


# ============================================================
# GENERATE ALL TEN COUNTERFACTUAL SCENARIOS
# ============================================================

print("\nGenerating common-population counterfactual scenarios...")

scenario_frames = []

scenario_columns = [
    "match_id",
    "match_date",
    "comp_name",
    "season",
    "competition_group",
    "legal_balls_remaining",
    "required_run_rate",
    "balls_band",
    "run_rate_band",
    "match_weight",
]

for wicket_count in range(10):

    scenario = common_test[scenario_columns].copy()

    # Change only the hypothetical wicket count.
    # Balls remaining, required run rate and competition stay fixed.
    scenario["wickets_lost"] = wicket_count

    scenario["predicted_chase_success"] = (
        model.predict_proba(
            scenario[FEATURES]
        )[:, 1]
    )

    scenario_frames.append(scenario)

scenarios = pd.concat(
    scenario_frames,
    ignore_index=True,
)


# ============================================================
# MATCH THE ZERO-WICKET BASELINE
# ============================================================

baseline = scenarios[
    scenarios["wickets_lost"] == 0
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

scenarios = scenarios.merge(
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

scenarios["probability_change_vs_zero"] = (
    scenarios["predicted_chase_success"]
    - scenarios["zero_wicket_probability"]
)


# ============================================================
# VERIFY IDENTICAL POPULATION
# ============================================================

population_check = (
    scenarios.groupby("wickets_lost")
    .agg(
        matches=("match_id", "nunique"),
        states=("match_id", "size"),
        total_weight=("match_weight", "sum"),
    )
    .reset_index()
)

print("\n--- IDENTICAL POPULATION CHECK ---")
print(
    population_check.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

if population_check["matches"].nunique() != 1:
    raise ValueError(
        "Different match counts across wicket scenarios."
    )

if population_check["states"].nunique() != 1:
    raise ValueError(
        "Different state counts across wicket scenarios."
    )

if not np.allclose(
    population_check["total_weight"].to_numpy(),
    population_check["total_weight"].iloc[0],
):
    raise ValueError(
        "Total weights differ across wicket scenarios."
    )


# ============================================================
# OVERALL MARGINAL EFFECT
# ============================================================

overall_rows = []

for wicket_count, group in scenarios.groupby("wickets_lost"):

    overall_rows.append({
        "wickets_lost": wicket_count,
        "supported_matches": group["match_id"].nunique(),
        "supported_states": len(group),
        "mean_predicted_success_probability": np.average(
            group["predicted_chase_success"],
            weights=group["match_weight"],
        ),
        "mean_probability_change_vs_zero": np.average(
            group["probability_change_vs_zero"],
            weights=group["match_weight"],
        ),
        "median_probability_change_vs_zero": (
            group["probability_change_vs_zero"].median()
        ),
    })

overall = pd.DataFrame(overall_rows)


# ============================================================
# EFFECT BY MATCH SITUATION
# ============================================================

situation_rows = []

for keys, group in scenarios.groupby(
    [
        "wickets_lost",
        "competition_group",
        "balls_band",
        "run_rate_band",
    ]
):

    wicket_count, comp_group, balls_group, rr_group = keys

    situation_rows.append({
        "wickets_lost": wicket_count,
        "competition_group": comp_group,
        "balls_band": balls_group,
        "run_rate_band": rr_group,
        "supported_matches": group["match_id"].nunique(),
        "supported_states": len(group),
        "mean_predicted_success_probability": np.average(
            group["predicted_chase_success"],
            weights=group["match_weight"],
        ),
        "mean_probability_change_vs_zero": np.average(
            group["probability_change_vs_zero"],
            weights=group["match_weight"],
        ),
    })

situation_summary = pd.DataFrame(situation_rows)


# ============================================================
# SAVE OUTPUTS
# ============================================================

overall.to_csv(
    OUTPUT_DIR / "wicket_marginal_effect_overall.csv",
    index=False,
)

situation_summary.to_csv(
    OUTPUT_DIR / "wicket_marginal_effect_by_situation.csv",
    index=False,
)

support.to_csv(
    OUTPUT_DIR / "wicket_marginal_effect_historical_support.csv",
    index=False,
)

support_diagnostics.to_csv(
    OUTPUT_DIR / "wicket_marginal_effect_common_support.csv",
    index=False,
)

population_check.to_csv(
    OUTPUT_DIR / "wicket_marginal_effect_population_check.csv",
    index=False,
)

scenarios.to_csv(
    OUTPUT_DIR / "wicket_marginal_effect_scenarios.csv",
    index=False,
)

# Save the exact common test population for reproducibility.
common_test.to_csv(
    OUTPUT_DIR / "wicket_marginal_effect_common_test_population.csv",
    index=False,
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 80)
print("COMMON-POPULATION WICKET MARGINAL EFFECT")
print("=" * 80)

print(
    overall.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\n--- POPULATION VERIFICATION ---")
print(population_check.to_string(index=False))

print("\n--- EFFECT BY MATCH SITUATION ---")

if not situation_summary.empty:
    print(
        situation_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
else:
    print("No situation-level results available.")

print("\n" + "=" * 80)
print(f"Outputs saved to: {OUTPUT_DIR}")
print("=" * 80)
