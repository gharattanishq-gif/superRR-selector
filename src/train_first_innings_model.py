
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==================================================
# 1. PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "outputs" / "processed" / "first_innings_states.csv"
OUTPUT = ROOT / "outputs" / "models"

OUTPUT.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATA
# ==================================================

print("\nLoading first-innings modelling data...")

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

print("Modelling rows:", len(df))
print("Unique matches:", df["match_id"].nunique())


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
target = "final_score"

X = df[features]
y = df[target]


# ==================================================
# 4. CHRONOLOGICAL MATCH-LEVEL SPLIT
# ==================================================

match_dates = (
    df.groupby("match_id")["match_date"]
    .min()
    .sort_values()
)

unique_matches = match_dates.index.tolist()

split_index = int(len(unique_matches) * 0.80)

train_matches = set(unique_matches[:split_index])
test_matches = set(unique_matches[split_index:])

train_mask = df["match_id"].isin(train_matches)
test_mask = df["match_id"].isin(test_matches)

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("\nCHRONOLOGICAL DATA SPLIT")
print("Training matches:", len(train_matches))
print("Testing matches:", len(test_matches))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

assert train_matches.isdisjoint(test_matches)


# ==================================================
# 5. PREPROCESSING
# ==================================================

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


# ==================================================
# 6. DEFINE MODELS
# ==================================================

models = {
    "Ridge Regression": Ridge(alpha=1.0),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
}


# ==================================================
# 7. TRAIN AND EVALUATE
# ==================================================

results = []
predictions = []

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    model_predictions = pd.DataFrame({
        "match_id": df.loc[test_mask, "match_id"].values,
        "match_date": df.loc[test_mask, "match_date"].values,
        "legal_balls_remaining": df.loc[
            test_mask, "legal_balls_remaining"
        ].values,
        "current_score": df.loc[
            test_mask, "current_score"
        ].values,
        "wickets_lost": df.loc[
            test_mask, "wickets_lost"
        ].values,
        "actual_final_score": y_test.values,
        "predicted_final_score": y_pred,
        "model": model_name
    })

    predictions.append(model_predictions)

    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2:   {r2:.4f}")


# ==================================================
# 8. COMPILE RESULTS
# ==================================================

results_df = pd.DataFrame(results).sort_values(
    by="MAE",
    ascending=True
)

predictions_df = pd.concat(
    predictions,
    ignore_index=True
)


# ==================================================
# 9. SAVE OUTPUTS
# ==================================================

results_path = OUTPUT / "first_innings_model_comparison.csv"
predictions_path = OUTPUT / "first_innings_test_predictions.csv"

results_df.to_csv(results_path, index=False)
predictions_df.to_csv(predictions_path, index=False)


# ==================================================
# 10. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 55)
print("FIRST-INNINGS MODEL COMPARISON")
print("=" * 55)

print(results_df.round(4).to_string(index=False))

print("\nBEST MODEL BY MAE:")
print(results_df.iloc[0]["model"])

print("\nPREDICTION RECORDS:", len(predictions_df))

print("\nFILES SAVED:")
print(results_path)
print(predictions_path)

print("\nMODEL TRAINING AND EVALUATION COMPLETE.")
