
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# 1. PATHS AND SETTINGS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "outputs" / "processed" / "first_innings_states.csv"
OUTPUT = ROOT / "outputs" / "models" / "final_calibration_experiment"

OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_ESTIMATORS = 300
INITIAL_TRAIN_FRACTION = 0.50
N_FOLDS = 5


# ==================================================
# 2. LOAD AND PREPARE DATA
# ==================================================

print("\nLoading modelling data...")

df = pd.read_csv(INPUT, low_memory=False)

df["match_date"] = pd.to_datetime(
    df["match_date"], errors="coerce"
)

df["season"] = df["season"].astype(str)
df["venue_id"] = df["venue_id"].astype(str)
df["comp_name"] = df["comp_name"].astype(str)

df = df.dropna(
    subset=[
        "match_id",
        "match_date",
        "current_score",
        "wickets_lost",
        "legal_balls_remaining",
        "final_score"
    ]
).copy()

df = df.reset_index(drop=True)

df["competition_type"] = np.where(
    df["comp_name"].str.contains("IPL", case=False, na=False),
    "IPL",
    "T20I"
)

print("Total rows:", len(df))
print("Total matches:", df["match_id"].nunique())


# ==================================================
# 3. FEATURES AND TARGET
# ==================================================

numeric_features = [
    "current_score",
    "wickets_lost",
    "legal_balls_remaining"
]

categorical_features = [
    "comp_name",
    "season",
    "venue_id"
]

features = numeric_features + categorical_features

X = df[features]
y = df["final_score"]


# ==================================================
# 4. CHRONOLOGICAL TRAIN / TEST SPLIT
# ==================================================

match_dates = (
    df.groupby("match_id")["match_date"]
    .min()
    .sort_values()
)

unique_matches = match_dates.index.tolist()

split_index = int(len(unique_matches) * 0.80)

train_matches = unique_matches[:split_index]
test_matches = unique_matches[split_index:]

train_mask = df["match_id"].isin(train_matches)
test_mask = df["match_id"].isin(test_matches)

train_df = df.loc[train_mask].copy()
test_df = df.loc[test_mask].copy()

assert set(train_matches).isdisjoint(set(test_matches))

print("\nCHRONOLOGICAL SPLIT")
print("Training matches:", len(train_matches))
print("Testing matches:", len(test_matches))


# ==================================================
# 5. PREPROCESSING FUNCTION
# ==================================================

def create_pipeline():

    numeric_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_preprocessor, numeric_features),
        ("categorical", categorical_preprocessor, categorical_features)
    ])

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


# ==================================================
# 6. CHRONOLOGICAL OUT-OF-FOLD PREDICTIONS
# ==================================================

print("\nGENERATING CHRONOLOGICAL OOF PREDICTIONS")

ordered_train_matches = (
    train_df.groupby("match_id")["match_date"]
    .min()
    .sort_values()
    .index.tolist()
)

initial_size = int(
    len(ordered_train_matches) * INITIAL_TRAIN_FRACTION
)

initial_matches = ordered_train_matches[:initial_size]
remaining_matches = ordered_train_matches[initial_size:]

folds = np.array_split(remaining_matches, N_FOLDS)

oof_predictions = []

for fold_number, validation_matches in enumerate(folds, start=1):

    validation_matches = list(validation_matches)

    if not validation_matches:
        continue

    # Only matches earlier than the validation period
    fold_train_matches = set(
        ordered_train_matches[:ordered_train_matches.index(
            validation_matches[0]
        )]
    )

    fold_validation_matches = set(validation_matches)

    fold_train = train_df[
        train_df["match_id"].isin(fold_train_matches)
    ]

    fold_validation = train_df[
        train_df["match_id"].isin(fold_validation_matches)
    ]

    print(f"\nFold {fold_number}")
    print("Training matches:", fold_train["match_id"].nunique())
    print("Validation matches:", fold_validation["match_id"].nunique())

    pipeline = create_pipeline()

    pipeline.fit(
        fold_train[features],
        fold_train["final_score"]
    )

    predictions = pipeline.predict(
        fold_validation[features]
    )

    fold_output = pd.DataFrame({
        "match_id": fold_validation["match_id"].values,
        "match_date": fold_validation["match_date"].values,
        "competition_type": fold_validation["competition_type"].values,
        "actual_final_score": fold_validation["final_score"].values,
        "predicted_final_score": predictions,
        "fold": fold_number
    })

    oof_predictions.append(fold_output)


oof_df = pd.concat(
    oof_predictions,
    ignore_index=True
)

oof_df["error"] = (
    oof_df["predicted_final_score"]
    - oof_df["actual_final_score"]
)

oof_df.to_csv(
    OUTPUT / "training_oof_predictions.csv",
    index=False
)

print("\nOOF PREDICTION SUMMARY")
print("OOF matches:", oof_df["match_id"].nunique())
print("OOF rows:", len(oof_df))


# ==================================================
# 7. ESTIMATE COMPETITION-SPECIFIC CORRECTIONS
# ==================================================

print("\nESTIMATING TRAINING-ONLY CORRECTIONS")

corrections = (
    oof_df.groupby("competition_type")
    .agg(
        oof_matches=("match_id", "nunique"),
        oof_observations=("error", "count"),
        estimated_bias=("error", "mean")
    )
    .reset_index()
)

# Correction is the negative of estimated prediction bias
corrections["correction"] = -corrections["estimated_bias"]

print(corrections.round(3).to_string(index=False))

corrections.to_csv(
    OUTPUT / "competition_calibration_corrections.csv",
    index=False
)


# ==================================================
# 8. TRAIN FINAL MODEL ON ALL TRAINING MATCHES
# ==================================================

print("\nTRAINING FINAL RANDOM FOREST MODEL")

final_model = create_pipeline()

final_model.fit(
    train_df[features],
    train_df["final_score"]
)

test_df = test_df.copy()

test_df["baseline_prediction"] = final_model.predict(
    test_df[features]
)


# ==================================================
# 9. APPLY TRAINING-ONLY CORRECTIONS
# ==================================================

correction_map = corrections.set_index(
    "competition_type"
)["correction"].to_dict()

test_df["calibration_correction"] = (
    test_df["competition_type"].map(correction_map).fillna(0)
)

test_df["adjusted_prediction"] = (
    test_df["baseline_prediction"]
    + test_df["calibration_correction"]
)

test_df["baseline_error"] = (
    test_df["baseline_prediction"] - test_df["final_score"]
)

test_df["adjusted_error"] = (
    test_df["adjusted_prediction"] - test_df["final_score"]
)


# ==================================================
# 10. OVERALL MODEL COMPARISON
# ==================================================

def calculate_metrics(actual, predicted):

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted),
        "Mean_Error": np.mean(predicted - actual)
    }


comparison = []

for model_name, prediction_column in [
    ("Original Random Forest", "baseline_prediction"),
    ("Competition-Calibrated RF", "adjusted_prediction")
]:

    metrics = calculate_metrics(
        test_df["final_score"],
        test_df[prediction_column]
    )

    metrics["model"] = model_name
    comparison.append(metrics)

comparison_df = pd.DataFrame(comparison)

comparison_df = comparison_df[
    ["model", "MAE", "RMSE", "R2", "Mean_Error"]
]

print("\nOVERALL TEST PERFORMANCE")
print(comparison_df.round(4).to_string(index=False))


# ==================================================
# 11. COMPETITION-WISE COMPARISON
# ==================================================

competition_results = []

for competition, group in test_df.groupby("competition_type"):

    for model_name, prediction_column in [
        ("Original Random Forest", "baseline_prediction"),
        ("Competition-Calibrated RF", "adjusted_prediction")
    ]:

        metrics = calculate_metrics(
            group["final_score"],
            group[prediction_column]
        )

        metrics["competition"] = competition
        metrics["model"] = model_name
        metrics["matches"] = group["match_id"].nunique()

        competition_results.append(metrics)

competition_df = pd.DataFrame(competition_results)

print("\nCOMPETITION-WISE TEST PERFORMANCE")
print(
    competition_df.round(4).to_string(index=False)
)


# ==================================================
# 12. MATCH-LEVEL COMPARISON
# ==================================================

match_results = []

for competition, group in test_df.groupby(
    ["competition_type", "match_id"]
):

    competition_type, match_id = competition

    for model_name, error_column in [
        ("Original Random Forest", "baseline_error"),
        ("Competition-Calibrated RF", "adjusted_error")
    ]:

        errors = group[error_column]

        match_results.append({
            "competition_type": competition_type,
            "match_id": match_id,
            "model": model_name,
            "match_bias": errors.mean(),
            "match_MAE": errors.abs().mean()
        })

match_df = pd.DataFrame(match_results)


# ==================================================
# 13. SAVE ALL RESULTS
# ==================================================

comparison_df.to_csv(
    OUTPUT / "overall_model_comparison.csv",
    index=False
)

competition_df.to_csv(
    OUTPUT / "competition_model_comparison.csv",
    index=False
)

match_df.to_csv(
    OUTPUT / "match_level_comparison.csv",
    index=False
)

test_df.to_csv(
    OUTPUT / "final_test_predictions.csv",
    index=False
)


# ==================================================
# 14. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 65)
print("FINAL CALIBRATION EXPERIMENT")
print("=" * 65)

print("\nEstimated corrections:")
print(corrections.round(3).to_string(index=False))

print("\nOverall performance:")
print(comparison_df.round(4).to_string(index=False))

print("\nCompetition performance:")
print(competition_df.round(4).to_string(index=False))

print("\nFiles saved in:")
print(OUTPUT)

print("\nCALIBRATION EXPERIMENT COMPLETE.")
