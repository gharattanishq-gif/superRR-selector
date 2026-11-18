
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "outputs" / "processed"
OUT = ROOT / "outputs" / "analysis" / "eligibility_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "batting": PROCESSED / "player_match_batting.csv",
    "bowling": PROCESSED / "player_match_bowling.csv",
    "pressure_batting": PROCESSED / "player_match_pressure_batting.csv",
    "pressure_bowling": PROCESSED / "player_match_pressure_bowling.csv",
    "matches": ROOT / "Data Pack" / "matches.csv",
}

for label, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} file: {path}")

bat = pd.read_csv(FILES["batting"])
bowl = pd.read_csv(FILES["bowling"])
pbat = pd.read_csv(FILES["pressure_batting"])
pbowl = pd.read_csv(FILES["pressure_bowling"])
matches = pd.read_csv(FILES["matches"])


def require(df, columns, label):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{label}: missing required columns {missing}")


require(
    bat,
    ["match_id", "striker_id", "striker_name",
     "match_format", "comp_name", "season"],
    "Batting stats"
)

require(
    bowl,
    ["match_id", "bowler_id", "bowler_name",
     "match_format", "comp_name", "season"],
    "Bowling stats"
)

require(
    pbat,
    ["match_id", "striker_id", "striker_name",
     "competition_group", "pressure_level", "pressure_deliveries"],
    "Batting pressure"
)

require(
    pbowl,
    ["match_id", "bowler_id", "bowler_name",
     "competition_group", "pressure_level", "pressure_deliveries"],
    "Bowling pressure"
)

require(matches, ["match_id", "comp_id"], "Matches")


def prep_base(df, role):
    player_col = "striker_id" if role == "Batting" else "bowler_id"
    name_col = "striker_name" if role == "Batting" else "bowler_name"

    d = df.copy()

    d["match_format"] = (
        d["match_format"].astype(str).str.strip().str.upper()
    )
    d["comp_name"] = d["comp_name"].astype(str).str.strip()
    d["season"] = d["season"].astype(str).str.strip()

    d = d[
        d[player_col].notna() & d["match_id"].notna()
    ].copy()

    # IPL: individual seasons.
    ipl = d[
        (d["match_format"] == "T20")
        & d["comp_name"].str.fullmatch(
            r"IPL\s+\d{4}", case=False, na=False
        )
    ].copy()

    # Senior men's T20Is: aggregate the complete evaluation window.
    t20i = d[d["match_format"] == "T20I"].copy()

    # Count each player-match only once.
    ipl = ipl.drop_duplicates(["match_id", player_col])
    t20i = t20i.drop_duplicates(["match_id", player_col])

    ipl_counts = (
        ipl.groupby(
            [player_col, name_col, "season"],
            dropna=False
        )
        .agg(
            matches_played=("match_id", "nunique"),
            first_match=(
                "match_date", "min"
            ) if "match_date" in ipl.columns
            else ("match_id", "min"),
            last_match=(
                "match_date", "max"
            ) if "match_date" in ipl.columns
            else ("match_id", "max"),
        )
        .reset_index()
    )

    ipl_counts["population"] = "IPL"
    ipl_counts["evaluation_unit"] = "player-season"

    t20i_counts = (
        t20i.groupby(
            [player_col, name_col],
            dropna=False
        )
        .agg(
            matches_played=("match_id", "nunique"),
            first_match=(
                "match_date", "min"
            ) if "match_date" in t20i.columns
            else ("match_id", "min"),
            last_match=(
                "match_date", "max"
            ) if "match_date" in t20i.columns
            else ("match_id", "max"),
        )
        .reset_index()
    )

    t20i_counts["season"] = "2022-2026 combined"
    t20i_counts["population"] = "Senior men's T20I"
    t20i_counts["evaluation_unit"] = (
        "player across evaluation window"
    )

    return (
        pd.concat(
            [ipl_counts, t20i_counts],
            ignore_index=True
        ),
        player_col
    )


def pressure_exposure(df, role):
    player_col = "striker_id" if role == "Batting" else "bowler_id"

    d = df.copy()

    d["competition_group"] = (
        d["competition_group"].astype(str).str.strip().str.upper()
    )
    d["pressure_level"] = (
        d["pressure_level"].astype(str).str.strip().str.title()
    )
    d["season"] = d["season"].astype(str).str.strip()

    d = d[
        d["competition_group"].isin(["IPL", "T20I"])
    ].copy()

    d["pressure_deliveries"] = pd.to_numeric(
        d["pressure_deliveries"],
        errors="coerce"
    ).fillna(0)

    ipl = d[d["competition_group"] == "IPL"].copy()
    t20i = d[d["competition_group"] == "T20I"].copy()

    def aggregate(frame, keys):
        return (
            frame.groupby(keys, dropna=False)
            .agg(
                pressure_matches=("match_id", "nunique"),
                pressure_deliveries=(
                    "pressure_deliveries", "sum"
                ),
                high_pressure_matches=(
                    "match_id",
                    lambda x: x[
                        frame.loc[
                            x.index, "pressure_level"
                        ].eq("High")
                    ].nunique()
                ),
                high_pressure_deliveries=(
                    "pressure_deliveries",
                    lambda x: x[
                        frame.loc[
                            x.index, "pressure_level"
                        ].eq("High")
                    ].sum()
                ),
            )
            .reset_index()
        )

    ipl_exp = aggregate(ipl, [player_col, "season"])
    ipl_exp["population"] = "IPL"

    t20i_exp = aggregate(t20i, [player_col])
    t20i_exp["population"] = "Senior men's T20I"
    t20i_exp["season"] = "2022-2026 combined"

    return pd.concat(
        [ipl_exp, t20i_exp],
        ignore_index=True
    )


def run_role(stats, pressure, role):
    base, player_col = prep_base(stats, role)
    exp = pressure_exposure(pressure, role)

    base = base.merge(
        exp,
        on=[player_col, "population", "season"],
        how="left"
    )

    base["pressure_window_coverage"] = (
        base["pressure_matches"].notna()
    )

    for col in [
        "pressure_matches",
        "pressure_deliveries",
        "high_pressure_matches",
        "high_pressure_deliveries",
    ]:
        base[col] = base[col].fillna(0).astype(int)

    population_thresholds = {
        "IPL": [8, 9, 10],
        "Senior men's T20I": [8, 12, 15],
    }

    summaries = []
    detail_frames = []

    for population, group in base.groupby("population"):

        for threshold in population_thresholds[population]:

            eligible = group[
                group["matches_played"] >= threshold
            ].copy()

            covered = eligible[
                eligible["pressure_window_coverage"]
            ]

            summaries.append({
                "role": role,
                "population": population,
                "threshold_matches": threshold,
                "eligible_units": len(eligible),
                "eligible_unique_players": (
                    eligible[player_col].nunique()
                ),
                "population_units": len(group),
                "population_unique_players": (
                    group[player_col].nunique()
                ),
                "eligible_unit_pct": round(
                    100 * len(eligible) / len(group), 2
                ) if len(group) else 0,
                "eligible_players_with_pressure_window_data": int(
                    eligible["pressure_window_coverage"].sum()
                ),
                "eligible_players_with_any_pressure_match": int(
                    (eligible["pressure_matches"] > 0).sum()
                ),
                "eligible_players_with_high_pressure_match": int(
                    (eligible["high_pressure_matches"] > 0).sum()
                ),
                "mean_pressure_matches_among_covered_units": (
                    round(covered["pressure_matches"].mean(), 2)
                    if len(covered) else 0
                ),
                "median_high_pressure_deliveries_among_covered_units": (
                    round(
                        covered["high_pressure_deliveries"].median(),
                        2
                    ) if len(covered) else 0
                ),
                "mean_high_pressure_deliveries_among_covered_units": (
                    round(
                        covered["high_pressure_deliveries"].mean(),
                        2
                    ) if len(covered) else 0
                ),
                "pressure_exposure_window": (
                    "2024 onward; earlier eligible matches "
                    "have no pressure-model coverage"
                ),
            })

            eligible["role"] = role
            eligible["threshold_matches"] = threshold

            detail_frames.append(eligible)

    return (
        pd.DataFrame(summaries),
        pd.concat(detail_frames, ignore_index=True)
        if detail_frames else pd.DataFrame(),
        base
    )


# Run batting and bowling separately.
bat_summary, bat_detail, bat_base = run_role(
    bat, pbat, "Batting"
)

bowl_summary, bowl_detail, bowl_base = run_role(
    bowl, pbowl, "Bowling"
)

summary = pd.concat(
    [bat_summary, bowl_summary],
    ignore_index=True
)

details = pd.concat(
    [bat_detail, bowl_detail],
    ignore_index=True
)

# Save threshold summary.
summary.to_csv(
    OUT / "eligibility_threshold_summary.csv",
    index=False
)

# Save player-level eligibility details.
details.to_csv(
    OUT / "eligible_player_details.csv",
    index=False
)

# Save the complete player population baseline.
pd.concat(
    [
        bat_base.assign(role="Batting"),
        bowl_base.assign(role="Bowling"),
    ],
    ignore_index=True
).to_csv(
    OUT / "player_population_baseline.csv",
    index=False
)

# Validation checks.
validation = pd.DataFrame([
    {
        "check": "Batting IPL player-match rows",
        "value": int(
            len(bat_base[bat_base.population == "IPL"])
        )
    },
    {
        "check": "Batting T20I player rows",
        "value": int(
            len(
                bat_base[
                    bat_base.population == "Senior men's T20I"
                ]
            )
        )
    },
    {
        "check": "Bowling IPL player-match rows",
        "value": int(
            len(bowl_base[bowl_base.population == "IPL"])
        )
    },
    {
        "check": "Bowling T20I player rows",
        "value": int(
            len(
                bowl_base[
                    bowl_base.population == "Senior men's T20I"
                ]
            )
        )
    },
    {
        "check": "Matches file unique match IDs",
        "value": int(matches.match_id.nunique())
    },
    {
        "check": "Batting stats match IDs absent from matches.csv",
        "value": int(
            (~bat.match_id.isin(matches.match_id)).sum()
        )
    },
    {
        "check": "Bowling stats match IDs absent from matches.csv",
        "value": int(
            (~bowl.match_id.isin(matches.match_id)).sum()
        )
    },
    {
        "check": "Pressure batting rows",
        "value": int(len(pbat))
    },
    {
        "check": "Pressure bowling rows",
        "value": int(len(pbowl))
    },
    {
        "check": "Pressure batting first date",
        "value": str(pbat.match_date.min())
    },
    {
        "check": "Pressure bowling first date",
        "value": str(pbowl.match_date.min())
    },
])

validation.to_csv(
    OUT / "eligibility_diagnostic_validation.csv",
    index=False
)

print("\nELIGIBILITY THRESHOLD SUMMARY")
print(summary.to_string(index=False))

print(f"\nSaved outputs to: {OUT}")

print(
    "\nNote: IPL thresholds are per season. "
    "T20I thresholds are aggregated across "
    "the full 2022-2026 evaluation window."
)

print(
    "Pressure exposure is reported only for the available "
    "pressure-model window (2024 onward); pre-2024 exposure "
    "is not assumed to be zero."
)
