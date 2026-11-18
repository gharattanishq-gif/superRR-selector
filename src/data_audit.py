
# ============================================================
# SUPERR SELECTOR - SEASON 2, ROUND 1
# File: data_audit.py
# Purpose: Initial data inspection and verification
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data Pack"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "audit"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("SUPERR SELECTOR - INITIAL DATA AUDIT")
print("=" * 60)

# ------------------------------------------------------------
# 2. VERIFY DATA DIRECTORY
# ------------------------------------------------------------

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data Pack not found: {DATA_DIR}")

csv_files = sorted(DATA_DIR.rglob("*.csv"))

print(f"\nCSV files found: {len(csv_files)}")

for file in csv_files:
    print(f"- {file.name}")

# ------------------------------------------------------------
# 3. CHECK LIBRARY VERSIONS
# ------------------------------------------------------------

print("\nPandas Version:", pd.__version__)
print("NumPy Version:", np.__version__)


# ------------------------------------------------------------
# 4. LOAD ALL DATASETS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LOADING DATASETS")
print("=" * 60)

datasets = {}

for file in csv_files:

    dataset_name = file.stem

    try:
        df = pd.read_csv(file)
        datasets[dataset_name] = df

        print(f"Loaded: {dataset_name}")

    except Exception as e:
        print(f"Error loading {file.name}: {e}")

competitions_df = datasets["competitions"]
matches_df = datasets["matches"]
teams_df = datasets["teams"]

# ------------------------------------------------------------
# 5. DATASET DIMENSIONS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET DIMENSIONS")
print("=" * 60)

for name, df in datasets.items():

    rows, columns = df.shape

    print(f"{name}: {rows:,} rows x {columns} columns")


# ------------------------------------------------------------
# 6. COLUMN INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("COLUMN INFORMATION")
print("=" * 60)

for name, df in datasets.items():

    print(f"\nDataset: {name}")
    print("-" * 50)

    print(df.dtypes)


# ------------------------------------------------------------
# 7. MISSING VALUE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing_records = []

for name, df in datasets.items():

    missing = df.isnull().sum()

    for column, count in missing.items():

        missing_records.append({
            "dataset": name,
            "column": column,
            "missing_count": count,
            "missing_percentage": round(
                (count / len(df)) * 100, 2
            ) if len(df) > 0 else 0
        })

    print(f"\n{name}: {missing.sum():,} missing values")


# Convert missing-value results into a DataFrame

missing_report = pd.DataFrame(missing_records)

missing_report = missing_report.sort_values(
    by="missing_count",
    ascending=False
)

# Save report

missing_report.to_csv(
    OUTPUT_DIR / "missing_values_report.csv",
    index=False
)

print("\nMissing-value report saved successfully.")


# ------------------------------------------------------------
# 8. DETAILED MISSING VALUE INSPECTION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TOP MISSING VALUE COLUMNS")
print("=" * 60)

for name, df in datasets.items():

    print(f"\nDATASET: {name}")
    print("-" * 60)

    missing = df.isnull().sum()

    missing = missing[missing > 0].sort_values(
        ascending=False
    )

    if missing.empty:
        print("No missing values.")
        continue

    report = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": (
            missing / len(df) * 100
        ).round(2)
    })

    print(report.to_string())


# ------------------------------------------------------------
# 9. DUPLICATE ROW INSPECTION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE ROW INSPECTION")
print("=" * 60)

for name, df in datasets.items():

    duplicate_count = df.duplicated().sum()

    print(
        f"{name}: {duplicate_count:,} "
        "completely duplicated rows"
    )


# ------------------------------------------------------------
# 10. PRIMARY KEY AND COMPOSITE KEY VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("KEY UNIQUENESS VALIDATION")
print("=" * 60)

key_definitions = {
    "competitions": ["comp_id"],
    "matches": ["match_id"],
    "teams": ["comp_id", "team_id"],
    "competition_team_player_mapping": [
        "comp_id", "team_id", "player_id"
    ],
    "playing_eleven": [
        "match_id", "team_id", "player_id"
    ],
    "ball_by_ball": [
        "match_id", "innings", "overid", "over_ball_num"
    ]
}

key_results = []

for dataset_name, key_columns in key_definitions.items():

    df = datasets[dataset_name]

    duplicate_keys = df.duplicated(
        subset=key_columns,
        keep=False
    )

    missing_keys = df[key_columns].isnull().any(axis=1)

    result = {
        "dataset": dataset_name,
        "key_columns": " + ".join(key_columns),
        "total_rows": len(df),
        "duplicate_key_rows": duplicate_keys.sum(),
        "missing_key_rows": missing_keys.sum()
    }

    key_results.append(result)

    print(f"\nDataset: {dataset_name}")
    print("Key:", " + ".join(key_columns))
    print("Duplicate key rows:", duplicate_keys.sum())
    print("Rows with missing key values:", missing_keys.sum())

    if duplicate_keys.any():
        print("\nSample duplicate keys:")
        print(
            df.loc[duplicate_keys, key_columns]
            .head(10)
            .to_string(index=False)
        )


# Save key validation report

key_report = pd.DataFrame(key_results)

key_report.to_csv(
    OUTPUT_DIR / "key_validation_report.csv",
    index=False
)

print("\nKey validation report saved.")


# ------------------------------------------------------------
# 11. PLAYING XI MISSINGNESS PATTERN
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PLAYING XI MISSINGNESS INVESTIGATION")
print("=" * 60)

playing_df = datasets["playing_eleven"]

columns_to_check = [
    "batting_order",
    "is_captain",
    "is_wk"
]

print("\nMissing values by is_played:")

print(
    playing_df.groupby("is_played")[columns_to_check]
    .apply(lambda group: group.isnull().sum())
    .to_string()
)

print("\nIs_played value counts:")

print(
    playing_df["is_played"]
    .value_counts(dropna=False)
    .to_string()
)

# ------------------------------------------------------------
# 12. INVESTIGATE PLAYING XI DUPLICATE KEYS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PLAYING XI DUPLICATE INVESTIGATION")
print("=" * 60)

playing_df = datasets["playing_eleven"]

duplicate_mask = playing_df.duplicated(
    subset=["match_id", "team_id", "player_id"],
    keep=False
)

duplicate_records = playing_df.loc[
    duplicate_mask
].sort_values(
    ["match_id", "team_id", "player_id"]
)

print("\nTotal duplicate-key records:", len(duplicate_records))

print("\nDetailed duplicate records:")

print(
    duplicate_records.to_string(index=False)
)

# Save for further investigation

duplicate_records.to_csv(
    OUTPUT_DIR / "playing_eleven_duplicate_records.csv",
    index=False
)


# ------------------------------------------------------------
# 13. INVESTIGATE DUPLICATE GROUP PATTERNS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE GROUP ANALYSIS")
print("=" * 60)

duplicate_groups = (
    duplicate_records.groupby(
        ["match_id", "team_id", "player_id"]
    )
    .agg(
        record_count=("is_played", "size"),
        played_values=("is_played", lambda x: list(x)),
        batting_orders=("batting_order", lambda x: list(x)),
        impact_values=("is_impact", lambda x: list(x)),
        impact_off_values=("is_impact_off", lambda x: list(x))
    )
    .reset_index()
)

print(duplicate_groups.to_string(index=False))

print("\nDuplicate groups by record count:")

print(
    duplicate_groups["record_count"]
    .value_counts()
    .sort_index()
    .to_string()
)

# ------------------------------------------------------------
# 14. CROSS-VERIFY PLAYING XI DUPLICATES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PLAYING XI vs BALL-BY-BALL VERIFICATION")
print("=" * 60)

duplicate_match_ids = duplicate_records["match_id"].unique()

ball_df = datasets["ball_by_ball"]
matches_df = datasets["matches"]

for match_id in duplicate_match_ids:

    print(f"\nMATCH ID: {match_id}")
    print("-" * 60)

    match_info = matches_df[
        matches_df["match_id"] == match_id
    ]

    print("Match information:")
    print(
        match_info[
            ["match_id", "comp_id", "match_date",
             "home_team", "away_team", "stage"]
        ].to_string(index=False)
    )

    match_duplicates = duplicate_records[
        duplicate_records["match_id"] == match_id
    ]

    match_deliveries = ball_df[
        ball_df["match_id"] == match_id
    ]

    for _, player in match_duplicates[
        ["team_id", "player_id"]
    ].drop_duplicates().iterrows():

        team_id = player["team_id"]
        player_id = player["player_id"]

        player_rows = match_duplicates[
            (match_duplicates["team_id"] == team_id) &
            (match_duplicates["player_id"] == player_id)
        ]

        player_deliveries = match_deliveries[
            (match_deliveries["batting_team_id"] == team_id) &
            (
                (match_deliveries["striker_id"] == player_id) |
                (match_deliveries["non_striker_id"] == player_id)
            )
        ]

        print(f"\nPlayer ID: {player_id}")
        print("Team ID:", team_id)
        print(
            "Recorded batting orders:",
            player_rows["batting_order"].tolist()
        )
        print(
            "Appeared in batting deliveries:",
            not player_deliveries.empty
        )

        if not player_deliveries.empty:

            first_appearance = player_deliveries.iloc[0]

            print(
                "First recorded batting order:",
                first_appearance["batting_order"]
            )

            print(
                "Striker in first appearance:",
                first_appearance["striker_id"] == player_id
            )

# ------------------------------------------------------------
# 15. INNINGS-LEVEL BATTING ORDER VERIFICATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("INNINGS-LEVEL BATTING ORDER VERIFICATION")
print("=" * 60)

for match_id in duplicate_match_ids:

    print(f"\nMATCH ID: {match_id}")
    print("-" * 60)

    match_duplicates = duplicate_records[
        duplicate_records["match_id"] == match_id
    ]

    match_deliveries = ball_df[
        ball_df["match_id"] == match_id
    ]

    for _, player in match_duplicates[
        ["team_id", "player_id"]
    ].drop_duplicates().iterrows():

        team_id = player["team_id"]
        player_id = player["player_id"]

        player_rows = match_duplicates[
            (match_duplicates["team_id"] == team_id) &
            (match_duplicates["player_id"] == player_id)
        ]

        player_deliveries = match_deliveries[
            (match_deliveries["batting_team_id"] == team_id) &
            (match_deliveries["striker_id"] == player_id)
        ].copy()

        print(f"\nPlayer: {player_id} | Team: {team_id}")
        print(
            "Playing XI batting orders:",
            player_rows["batting_order"].tolist()
        )

        if player_deliveries.empty:
            print("No striker deliveries found.")
            continue

        player_deliveries = player_deliveries.sort_values(
            ["innings", "innings_ball_num", "over_ball_num"]
        )

        print("Ball-by-ball batting order by innings:")

        print(
            player_deliveries.groupby("innings")[
                "batting_order"
            ].agg(lambda x: sorted(x.unique())).to_dict()
        )

# ------------------------------------------------------------
# 16. FULL PLAYING XI vs BALL-BY-BALL BATTING ORDER AUDIT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FULL BATTING ORDER CONSISTENCY AUDIT")
print("=" * 60)

# Extract unique batting orders from striker deliveries
bb_order = (
    ball_df.groupby(
        ["match_id", "batting_team_id", "striker_id"]
    )["batting_order"]
    .agg(
        bb_batting_order=lambda x: sorted(x.dropna().unique()),
        striker_deliveries="count"
    )
    .reset_index()
)

bb_order = bb_order.rename(columns={
    "batting_team_id": "team_id",
    "striker_id": "player_id"
})

# Merge with playing XI records
playing_order_audit = playing_df.merge(
    bb_order,
    on=["match_id", "team_id", "player_id"],
    how="left"
)

# Classify records
playing_order_audit["order_status"] = "Not verified - no striker delivery"

has_bb_order = playing_order_audit["bb_batting_order"].notna()

playing_order_audit.loc[
    has_bb_order & playing_order_audit["batting_order"].isna(),
    "order_status"
] = "Missing in playing XI"

playing_order_audit.loc[
    has_bb_order &
    playing_order_audit["batting_order"].notna() &
    playing_order_audit.apply(
        lambda row: row["batting_order"] in row["bb_batting_order"]
        if isinstance(row["bb_batting_order"], list) else False,
        axis=1
    ),
    "order_status"
] = "Match"

playing_order_audit.loc[
    has_bb_order &
    playing_order_audit["batting_order"].notna() &
    (playing_order_audit["order_status"] != "Match"),
    "order_status"
] = "Mismatch"

# Summary
print("\nBatting-order audit summary:")
print(playing_order_audit["order_status"].value_counts(dropna=False))

# Save detailed report
playing_order_audit.to_csv(
    OUTPUT_DIR / "playing_xi_batting_order_audit.csv",
    index=False
)

print("\nDetailed audit saved successfully.")


# ------------------------------------------------------------
# 17. INSPECT BATTING ORDER EXCEPTIONS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("BATTING ORDER EXCEPTIONS")
print("=" * 60)

exceptions = playing_order_audit[
    playing_order_audit["order_status"].isin(
        ["Mismatch", "Missing in playing XI"]
    )
].copy()

exception_columns = [
    "match_id",
    "team_id",
    "player_id",
    "batting_order",
    "bb_batting_order",
    "striker_deliveries",
    "order_status"
]

print(exceptions[exception_columns].to_string(index=False))

print("\nUnique affected players:", 
      exceptions[["match_id", "team_id", "player_id"]]
      .drop_duplicates().shape[0])

exceptions.to_csv(
    OUTPUT_DIR / "batting_order_exceptions.csv",
    index=False
)

# ------------------------------------------------------------
# 18. MATCH-LEVEL BATTING ORDER DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MATCH-LEVEL BATTING ORDER DISTRIBUTION")
print("=" * 60)

for match_id in [13613, 21176, 28962]:

    print(f"\nMATCH ID: {match_id}")
    print("-" * 60)

    match_data = ball_df[
        ball_df["match_id"] == match_id
    ]

    batting_summary = (
        match_data.groupby(
            ["innings", "batting_team_id", "striker_id"]
        )
        .agg(
            player_name=("striker_name", "first"),
            batting_order=("batting_order", "first"),
            unique_orders=("batting_order", "nunique"),
            deliveries=("striker_id", "size")
        )
        .reset_index()
        .sort_values(["innings", "batting_order"])
    )

    print(batting_summary.to_string(index=False))

# ============================================================
# FOREIGN KEY AND RELATIONSHIP VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("FOREIGN KEY AND RELATIONSHIP VALIDATION")
print("=" * 60)

# 1. Match -> Competition
invalid_match_comp = matches_df[
    ~matches_df["comp_id"].isin(competitions_df["comp_id"])
]

print("\n1. Matches with invalid competition IDs:")
print(len(invalid_match_comp))


# 2. Ball-by-ball -> Match
invalid_bb_match = ball_df[
    ~ball_df["match_id"].isin(matches_df["match_id"])
]

print("\n2. Ball-by-ball records with invalid match IDs:")
print(len(invalid_bb_match))


# 3. Playing XI -> Match
invalid_xi_match = playing_df[
    ~playing_df["match_id"].isin(matches_df["match_id"])
]

print("\n3. Playing XI records with invalid match IDs:")
print(len(invalid_xi_match))


# 4. Match teams -> Competition teams
match_teams = pd.concat([
    matches_df[["comp_id", "home_team_id"]]
        .rename(columns={"home_team_id": "team_id"}),

    matches_df[["comp_id", "away_team_id"]]
        .rename(columns={"away_team_id": "team_id"})
]).drop_duplicates()

valid_teams = teams_df[["comp_id", "team_id"]].drop_duplicates()

team_check = match_teams.merge(
    valid_teams,
    on=["comp_id", "team_id"],
    how="left",
    indicator=True
)

invalid_match_teams = team_check[
    team_check["_merge"] == "left_only"
]

print("\n4. Match teams missing from teams dataset:")
print(len(invalid_match_teams))

if not invalid_match_teams.empty:
    print(invalid_match_teams)


# 5. Ball-by-ball teams -> Match participants
match_team_lookup = matches_df[
    ["match_id", "home_team_id", "away_team_id"]
]

bb_team_check = ball_df.merge(
    match_team_lookup,
    on="match_id",
    how="left"
)

valid_batting_team = (
    (bb_team_check["batting_team_id"] == bb_team_check["home_team_id"]) |
    (bb_team_check["batting_team_id"] == bb_team_check["away_team_id"])
)

valid_bowling_team = (
    (bb_team_check["bowling_team_id"] == bb_team_check["home_team_id"]) |
    (bb_team_check["bowling_team_id"] == bb_team_check["away_team_id"])
)

invalid_bb_teams = bb_team_check[
    ~valid_batting_team | ~valid_bowling_team
]

print("\n5. Ball-by-ball records with invalid participating teams:")
print(len(invalid_bb_teams))

if not invalid_bb_teams.empty:
    print(
        invalid_bb_teams[
            [
                "match_id",
                "innings",
                "batting_team_id",
                "bowling_team_id",
                "home_team_id",
                "away_team_id"
            ]
        ].drop_duplicates().head(20)
    )

print("\nForeign-key validation completed.")