
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# SUPERR SELECTOR — FINAL RANKINGS VALIDATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "outputs" / "analysis"
COMBINED_DIR = ANALYSIS_DIR / "combined_player_scores"
SHORTLIST_DIR = ANALYSIS_DIR / "final_shortlists"
REPORT_DIR = ANALYSIS_DIR / "final_validation"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

results = []


# ============================================================
# GENERAL HELPERS
# ============================================================

def record(status, check, details=""):
    results.append({
        "status": status,
        "check": check,
        "details": str(details)
    })

    symbol = {
        "PASS": "[PASS]",
        "FAIL": "[FAIL]",
        "WARNING": "[WARNING]"
    }.get(status, "[INFO]")

    print(f"{symbol} {check}: {details}")


def load_csv(path, label):
    if not path.exists():
        record("FAIL", label, f"File not found: {path}")
        return None

    try:
        df = pd.read_csv(path)
        record(
            "PASS",
            label,
            f"Loaded {len(df)} rows × {len(df.columns)} columns from {path}"
        )
        return df
    except Exception as exc:
        record("FAIL", label, f"Could not load file: {exc}")
        return None


def check_columns(df, required, label):
    missing = [col for col in required if col not in df.columns]

    if missing:
        record("FAIL", label, f"Missing columns: {missing}")
        return False

    record("PASS", label, "All required columns found.")
    return True


def check_duplicates(df, keys, label):
    if not all(col in df.columns for col in keys):
        record("FAIL", label, f"Cannot check duplicate keys: {keys}")
        return

    count = int(df.duplicated(subset=keys, keep=False).sum())

    if count:
        record(
            "FAIL",
            label,
            f"{count} duplicate rows using {keys}"
        )
    else:
        record(
            "PASS",
            label,
            f"0 duplicate rows using {keys}"
        )


def numeric_series(df, column):
    return pd.to_numeric(df[column], errors="coerce")



def check_score_range(df, column, label):
    if column not in df.columns:
        record("FAIL", label, f"Column not found: {column}")
        return

    values = pd.to_numeric(df[column], errors="coerce")

    # Missing scores are permitted when a component is unavailable.
    # Only validate scores that are actually present.
    available = values.notna()

    invalid = available & ~values.between(0, 100)

    count = int(invalid.sum())
    missing = int((~available).sum())

    if count:
        record(
            "FAIL",
            label,
            f"{count} available scores outside 0–100; "
            f"{missing} missing scores excluded."
        )
    else:
        record(
            "PASS",
            label,
            f"0 available scores outside 0–100; "
            f"{missing} missing scores appropriately excluded."
        )


def check_descending(df, column, label):
    if column not in df.columns:
        record("FAIL", label, f"Column not found: {column}")
        return

    values = numeric_series(df, column)

    if values.isna().any():
        record("FAIL", label, "Missing or non-numeric ranking scores.")
        return

    if values.is_monotonic_decreasing:
        record("PASS", label, "Scores are descending.")
    else:
        record("FAIL", label, "Scores are not sorted in descending order.")


def check_rank_sequence(df, label):
    if "rank" not in df.columns:
        record("FAIL", label, "Rank column not found.")
        return

    ranks = pd.to_numeric(df["rank"], errors="coerce")

    expected = list(range(1, len(df) + 1))
    observed = ranks.tolist()

    if ranks.isna().any() or observed != expected:
        record(
            "FAIL",
            label,
            f"Expected ranks {expected}; observed {observed}"
        )
    else:
        record("PASS", label, f"Observed ranks: {observed}")


def check_formula(df, actual_col, formula, label, mask=None):
    if actual_col not in df.columns:
        record("FAIL", label, f"Column not found: {actual_col}")
        return

    try:
        expected = formula(df)
        actual = numeric_series(df, actual_col)

        valid = actual.notna() & expected.notna()

        if mask is not None:
            valid &= mask

        mismatches = int(
            (~np.isclose(
                actual[valid],
                expected[valid],
                rtol=0,
                atol=0.011
            )).sum()
        )

        if mismatches:
            record(
                "FAIL",
                label,
                f"{mismatches} formula mismatches among {int(valid.sum())} scored rows."
            )
        else:
            record(
                "PASS",
                label,
                f"0 mismatches among {int(valid.sum())} scored rows."
            )

    except Exception as exc:
        record("FAIL", label, f"Formula validation error: {exc}")


# ============================================================
# COMBINED BATTERING / BOWLING VALIDATION
# ============================================================

def validate_combined(path, label):
    df = load_csv(path, label)

    if df is None:
        return

    required = [
        "player_id",
        "population",
        "period",
        "combined_score",
        "overall_index",
        "pressure_index",
        "overall_available",
        "pressure_available",
        "player_name_overall",
        "player_name_pressure"
    ]

    if not check_columns(df, required, f"{label} required columns"):
        return

    check_duplicates(
        df,
        ["player_id", "population", "period"],
        f"{label} duplicate key check"
    )

    check_score_range(df, "combined_score", f"{label} combined_score range")
    check_score_range(df, "overall_index", f"{label} overall_index range")
    check_score_range(df, "pressure_index", f"{label} pressure_index range")

    overall_available = (
        df["overall_available"].astype(str).str.lower().isin(["true", "1"])
    )

    pressure_available = (
        df["pressure_available"].astype(str).str.lower().isin(["true", "1"])
    )

    scored = (
        overall_available
        & pressure_available
        & df["overall_index"].notna()
        & df["pressure_index"].notna()
        & df["combined_score"].notna()
    )

    check_formula(
        df,
        "combined_score",
        lambda x: (
            0.35 * numeric_series(x, "overall_index")
            + 0.65 * numeric_series(x, "pressure_index")
        ),
        f"{label} 35:65 combined score formula",
        mask=scored
    )

    inconsistent = (
        (pressure_available & df["pressure_index"].isna())
        | (~pressure_available & df["pressure_index"].notna())
    )

    count = int(inconsistent.sum())

    if count:
        record(
            "FAIL",
            f"{label} pressure availability consistency",
            f"{count} inconsistent rows."
        )
    else:
        record(
            "PASS",
            f"{label} pressure availability consistency",
            "0 inconsistent rows."
        )

    if "player_name" in df.columns:
        name_check = (
            df["player_name_overall"].notna()
            & df["player_name"].notna()
        )

        mismatches = (
            df.loc[name_check, "player_name_overall"].astype(str).str.strip()
            != df.loc[name_check, "player_name"].astype(str).str.strip()
        ).sum()

        if mismatches:
            record(
                "FAIL",
                f"{label} player_name_overall consistency",
                f"{mismatches} name mismatches."
            )
        else:
            record(
                "PASS",
                f"{label} player_name_overall consistency",
                "0 name mismatches."
            )

        name_check = (
            df["player_name_pressure"].notna()
            & df["player_name"].notna()
        )

        mismatches = (
            df.loc[name_check, "player_name_pressure"].astype(str).str.strip()
            != df.loc[name_check, "player_name"].astype(str).str.strip()
        ).sum()

        if mismatches:
            record(
                "FAIL",
                f"{label} player_name_pressure consistency",
                f"{mismatches} name mismatches."
            )
        else:
            record(
                "PASS",
                f"{label} player_name_pressure consistency",
                "0 name mismatches."
            )


# ============================================================
# CROSS-COMPETITION VALIDATION
# ============================================================

def validate_cross_competition(path):
    df = load_csv(path, "cross_competition_rankings.csv")

    if df is None:
        return

    required = [
        "player_id",
        "index_type",
        "period",
        "cross_competition_score",
        "combined_score",
        "t20i_combined_score"
    ]

    if not check_columns(
        df,
        required,
        "Cross-competition required columns"
    ):
        return

    check_duplicates(
        df,
        ["player_id", "index_type", "period"],
        "Cross-competition duplicate key check"
    )

    check_score_range(
        df,
        "cross_competition_score",
        "Cross-competition score range"
    )

    check_formula(
        df,
        "cross_competition_score",
        lambda x: (
            0.65 * numeric_series(x, "combined_score")
            + 0.35 * numeric_series(x, "t20i_combined_score")
        ),
        "65:35 cross-competition formula"
    )


# ============================================================
# TOP 10 SHORTLIST VALIDATION
# ============================================================

def validate_basic_top10(filename, score_column):
    path = SHORTLIST_DIR / filename
    df = load_csv(path, filename)

    if df is None:
        return

    required = ["player_id", "rank", score_column]

    if not check_columns(df, required, f"{filename} required columns"):
        return

    if len(df) == 10:
        record("PASS", f"{filename} Top 10 row count", "Exactly 10 rows.")
    else:
        record(
            "FAIL",
            f"{filename} Top 10 row count",
            f"Expected 10 rows; found {len(df)}."
        )

    check_duplicates(
        df,
        ["player_id"],
        f"{filename} duplicate key check"
    )

    check_rank_sequence(df, f"{filename} rank sequence")
    check_score_range(df, score_column, f"{filename} score range")
    check_descending(df, score_column, f"{filename} score ordering")


def validate_overall_top10(filename):
    path = SHORTLIST_DIR / filename
    df = load_csv(path, filename)

    if df is None:
        return

    required = [
        "player_id",
        "rank",
        "ipl_wide_score",
        "t20i_score",
        "overall_cross_competition_score"
    ]

    if not check_columns(df, required, f"{filename} required columns"):
        return

    if len(df) == 10:
        record("PASS", f"{filename} Top 10 row count", "Exactly 10 rows.")
    else:
        record(
            "FAIL",
            f"{filename} Top 10 row count",
            f"Expected 10 rows; found {len(df)}."
        )

    check_duplicates(
        df,
        ["player_id"],
        f"{filename} duplicate key check"
    )

    check_rank_sequence(df, f"{filename} rank sequence")

    # Component scores are inputs to the weighted score.
    # They are NOT the ranking order of this shortlist.
    check_score_range(
        df,
        "ipl_wide_score",
        f"{filename} IPL-wide score range"
    )

    check_score_range(
        df,
        "t20i_score",
        f"{filename} T20I score range"
    )

    check_score_range(
        df,
        "overall_cross_competition_score",
        f"{filename} overall score range"
    )

    # Only the final weighted score must be descending.
    check_descending(
        df,
        "overall_cross_competition_score",
        f"{filename} overall score ordering"
    )

    check_formula(
        df,
        "overall_cross_competition_score",
        lambda x: (
            0.65 * numeric_series(x, "ipl_wide_score")
            + 0.35 * numeric_series(x, "t20i_score")
        ),
        f"{filename} 65:35 weighted score formula"
    )


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():
    print("\n" + "=" * 65)
    print("SUPERR SELECTOR — FINAL RANKINGS VALIDATION")
    print("=" * 65)

    validate_combined(
        COMBINED_DIR / "combined_batting_scores_by_competition.csv",
        "combined_batting_scores_by_competition.csv"
    )

    validate_combined(
        COMBINED_DIR / "combined_bowling_scores_by_competition.csv",
        "combined_bowling_scores_by_competition.csv"
    )

    validate_cross_competition(
        COMBINED_DIR / "cross_competition_rankings.csv"
    )

    # IPL-wide lists
    validate_basic_top10(
        "ipl_wide_batting_top10.csv",
        "ipl_wide_score"
    )

    validate_basic_top10(
        "ipl_wide_bowling_top10.csv",
        "ipl_wide_score"
    )

    # T20I lists
    validate_basic_top10(
        "t20i_batting_top10.csv",
        "t20i_score"
    )

    validate_basic_top10(
        "t20i_bowling_top10.csv",
        "t20i_score"
    )

    # Overall cross-competition lists
    validate_overall_top10(
        "overall_cross_comp_batting_top10.csv"
    )

    validate_overall_top10(
        "overall_cross_comp_bowling_top10.csv"
    )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report = pd.DataFrame(results)

    report_path = REPORT_DIR / "final_validation_report.csv"
    report.to_csv(report_path, index=False)

    passed = sum(r["status"] == "PASS" for r in results)
    warnings = sum(r["status"] == "WARNING" for r in results)
    failed = sum(r["status"] == "FAIL" for r in results)
    total = len(results)

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    print(f"Passed checks : {passed}")
    print(f"Warnings      : {warnings}")
    print(f"Failed checks : {failed}")
    print(f"Total checks  : {total}")

    print("\nDetailed report saved to:")
    print(report_path)

    if failed == 0 and warnings == 0:
        print("\nRESULT: ALL VALIDATION CHECKS PASSED.")
    elif failed == 0:
        print("\nRESULT: VALIDATION COMPLETED WITH WARNINGS.")
    else:
        print("\nRESULT: VALIDATION FAILED — resolve failed checks.")

    print("=" * 65)


if __name__ == "__main__":
    main()
