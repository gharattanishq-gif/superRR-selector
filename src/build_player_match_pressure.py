from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data Pack"
PROC = ROOT / "outputs" / "processed"
CAL = ROOT / "outputs" / "analysis" / "pressure_calibration"

PRESSURE = PROC / "pressure_states.csv"
BALLS = DATA / "ball_by_ball.csv"
MATCHES = DATA / "matches.csv"
POOLED = CAL / "pressure_threshold_comparison.csv"
PHASES = CAL / "pressure_classification_sensitivity_thresholds.csv"

OUT_BAT = PROC / "player_match_pressure_batting.csv"
OUT_BOWL = PROC / "player_match_pressure_bowling.csv"
OUT_CHECK = PROC / "player_match_pressure_validation.csv"

KEY = ["match_id", "innings", "overid", "over_ball_num"]
BOWLER_WICKETS = {"caught", "bowled", "lbw", "stump", "hit wicket"}

# Previously agreed policy: True means use phase-specific 40/75 thresholds.
PHASE_POLICY = {
    (1, "IPL", "Powerplay"): True, (1, "IPL", "Middle"): False,
    (1, "IPL", "Death"): True,
    (1, "T20I", "Powerplay"): True, (1, "T20I", "Middle"): False,
    (1, "T20I", "Death"): False,
    (2, "IPL", "Powerplay"): True, (2, "IPL", "Middle"): False,
    (2, "IPL", "Death"): True,
    (2, "T20I", "Powerplay"): True, (2, "T20I", "Middle"): False,
    (2, "T20I", "Death"): True,
}


def need(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


def num(s):
    return pd.to_numeric(s, errors="coerce")


def clean(s):
    return s.fillna("").astype(str).str.strip().str.lower()


def thresholds():
    p = pd.read_csv(POOLED)
    q = pd.read_csv(PHASES)
    p = p[p["scheme"].astype(str) == "40_75"].copy()
    p["innings"] = num(p["innings"]).astype(int)
    q["innings"] = num(q["innings"]).astype(int)

    pool = {
        (int(r.innings), str(r.competition_group), str(r.role)):
        (float(r.low_threshold), float(r.high_threshold))
        for r in p.itertuples()
    }
    phase = {
        (int(r.innings), str(r.competition_group), str(r.role), str(r.phase)):
        (float(r.phase_low_threshold), float(r.phase_high_threshold))
        for r in q.itertuples()
    }
    expected_pool = {
        (i, c, r) for i in (1, 2) for c in ("IPL", "T20I")
        for r in ("Batting", "Bowling")
    }
    expected_phase = {
        (i, c, r, ph) for i in (1, 2) for c in ("IPL", "T20I")
        for r in ("Batting", "Bowling")
        for ph in ("Powerplay", "Middle", "Death")
    }
    if set(pool) != expected_pool or set(phase) != expected_phase:
        raise ValueError("Calibration files do not contain all expected 40/75 thresholds.")
    return pool, phase


def get_phase(remaining, limit):
    if pd.isna(remaining) or pd.isna(limit) or limit <= 0:
        return np.nan
    progress = np.clip(1 - float(remaining) / float(limit), 0, 1)
    if progress < .25:
        return "Powerplay"
    if progress < .75:
        return "Middle"
    return "Death"


def classify(df, role, pool, phase):
    col = "batting_pressure" if role == "Batting" else "bowling_pressure"
    lows, highs, labels, sources = [], [], [], []
    for r in df.itertuples():
        i, comp, ph = int(r.innings), str(r.competition_group), str(r.phase)
        use_phase = PHASE_POLICY.get((i, comp, ph), False)
        if use_phase:
            lo, hi = phase[(i, comp, role, ph)]
            source = "phase_specific"
        else:
            lo, hi = pool[(i, comp, role)]
            source = "pooled"
        value = getattr(r, col)
        label = np.nan if pd.isna(value) else (
            "Low" if value < lo else "Moderate" if value <= hi else "High"
        )
        lows.append(lo); highs.append(hi); labels.append(label); sources.append(source)
    out = df.copy()
    out["pressure_level"] = labels
    out["pressure_low_threshold"] = lows
    out["pressure_high_threshold"] = highs
    out["threshold_type"] = sources
    return out


def main():
    for f in (PRESSURE, BALLS, MATCHES, POOLED, PHASES):
        if not f.exists():
            raise FileNotFoundError(f"Missing input: {f}")

    p = pd.read_csv(PRESSURE, low_memory=False)
    b = pd.read_csv(BALLS, low_memory=False)
    m = pd.read_csv(MATCHES, low_memory=False)

    need(p, KEY + [
        "source_row_id", "match_date", "comp_name", "season",
        "competition_group", "striker_id", "striker_name", "bowler_id",
        "bowler_name", "batting_team_id", "bowling_team_id", "runs",
        "is_wicket", "pressure_eligible", "batting_pressure",
        "bowling_pressure", "legal_balls_remaining",
    ], "pressure_states")
    need(b, KEY + [
        "runs", "is_four", "is_six", "is_wicket", "dismissal_type",
        "dismissed_player_id", "extras_type", "extras1_runs", "extras_2_runs",
    ], "ball_by_ball")
    need(m, ["match_id"], "matches")

    # Join by the actual delivery identity, never by source_row_id.
    for c in KEY:
        p[c] = num(p[c])
        b[c] = num(b[c])
    if b.duplicated(KEY).any():
        raise ValueError("Original ball_by_ball delivery key is not unique.")
    src_cols = KEY + [
        "runs", "is_four", "is_six", "is_wicket", "dismissal_type",
        "dismissed_player_id", "extras_type", "extras1_runs", "extras_2_runs",
    ]
    d = p.merge(
        b[src_cols], on=KEY, how="left", validate="many_to_one",
        suffixes=("_state", "_src"), indicator=True
    )
    unmatched = int((d["_merge"] != "both").sum())
    if unmatched:
        raise ValueError(f"{unmatched} pressure deliveries did not match source deliveries.")
    if (num(d["runs_state"]) != num(d["runs_src"])).fillna(True).any():
        raise ValueError("Runs mismatch after delivery-key join; stopping.")
    if (d["is_wicket_state"] != d["is_wicket_src"]).fillna(True).any():
        raise ValueError("Wicket mismatch after delivery-key join; stopping.")

    d["runs"] = num(d["runs_src"]).fillna(0)
    d["is_wicket"] = d["is_wicket_src"].fillna(False).astype(bool)
    d["extras_clean"] = clean(d["extras_type"])
    e1 = num(d["extras1_runs"]).fillna(0)
    e2 = num(d["extras_2_runs"]).fillna(0)

    missing_type_extras = d["extras_clean"].eq("") & ((e1 != 0) | (e2 != 0))
    ambiguous_type = d["extras_clean"].isin(["other", "penalty"])
    unknown_type = ~d["extras_clean"].isin(["", "wide", "overstep", "byes", "legbyes"])
    d["bowling_unresolved"] = missing_type_extras | ambiguous_type | unknown_type | e2.ne(0)

    # Project rule: wides and oversteps/no-balls are not legal balls.
    d["legal_delivery"] = ~d["extras_clean"].isin(["wide", "overstep"])
    d.loc[d["bowling_unresolved"], "legal_delivery"] = False

    # Match file stores revised innings limits as legal-ball counts; standard
    # T20 limit is 120 when no shortened limit is recorded.
    for c in ("first_inn_revised_overs", "second_inn_revised_overs"):
        if c not in m.columns:
            m[c] = np.nan
        m[c] = num(m[c])
    m = m[["match_id", "first_inn_revised_overs", "second_inn_revised_overs"]].drop_duplicates("match_id")
    d = d.merge(m, on="match_id", how="left", validate="many_to_one")
    lim1 = num(d["first_inn_revised_overs"]).fillna(120)
    lim2 = num(d["second_inn_revised_overs"]).fillna(120)
    d["scheduled_balls"] = np.where(num(d["innings"]) == 1, lim1, lim2)
    d["phase"] = [
        get_phase(rem, lim) for rem, lim in
        zip(d["legal_balls_remaining"], d["scheduled_balls"])
    ]

    d = d[d["competition_group"].isin(["IPL", "T20I"])].copy()
    d["innings"] = num(d["innings"]).astype(int)
    pool, phase = thresholds()

    # BATTERS: all eligible pressure deliveries; ambiguous extras do not alter
    # batter runs, but the ambiguity is exposed in the output.
    bat = d[d["pressure_eligible"].fillna(False) & d["batting_pressure"].notna()].copy()
    bat = classify(bat, "Batting", pool, phase)
    bat["balls_faced_value"] = bat["legal_delivery"].astype(int)
    bat["four_value"] = bat["is_four"].fillna(False).astype(bool).astype(int)
    bat["six_value"] = bat["is_six"].fillna(False).astype(bool).astype(int)
    bat["dismissal_value"] = (
        bat["is_wicket"] & bat["dismissed_player_id"].notna()
        & (num(bat["dismissed_player_id"]) == num(bat["striker_id"]))
    ).astype(int)

    bk = [
        "match_id", "match_date", "comp_name", "season", "competition_group",
        "innings", "phase", "striker_id", "striker_name", "batting_team_id",
        "pressure_level", "threshold_type",
    ]
    bout = bat.groupby(bk, dropna=False, as_index=False).agg(
        pressure_deliveries=("source_row_id", "size"),
        balls_faced=("balls_faced_value", "sum"),
        batter_runs=("runs", "sum"),
        fours=("four_value", "sum"),
        sixes=("six_value", "sum"),
        dismissals=("dismissal_value", "sum"),
        ambiguous_extras_deliveries=("bowling_unresolved", "sum"),
    )
    bout["strike_rate"] = np.where(
        bout["balls_faced"] > 0, bout["batter_runs"] * 100 / bout["balls_faced"], np.nan
    )
    bout["batting_stats_complete"] = bout["ambiguous_extras_deliveries"].eq(0)

    # BOWLERS: ambiguous extras are excluded from runs conceded and legal-ball
    # denominator. Byes/leg-byes are legal but are not charged to the bowler.
    bowl = d[d["pressure_eligible"].fillna(False) & d["bowling_pressure"].notna()].copy()
    bowl = classify(bowl, "Bowling", pool, phase)
    bowl["bowling_valid"] = ~bowl["bowling_unresolved"]
    bowl["bowling_ball_value"] = (bowl["legal_delivery"] & bowl["bowling_valid"]).astype(int)
    bowl["conceded_value"] = np.where(
        bowl["bowling_valid"],
        bowl["runs"] + np.where(bowl["extras_clean"].isin(["wide", "overstep"]), e1.loc[bowl.index], 0),
        np.nan,
    )
    dismissals = clean(bowl["dismissal_type"])
    bowl["wicket_value"] = (
        bowl["bowling_valid"] & bowl["is_wicket"] & dismissals.isin(BOWLER_WICKETS)
    ).astype(int)

    wok = [
        "match_id", "match_date", "comp_name", "season", "competition_group",
        "innings", "phase", "bowler_id", "bowler_name", "bowling_team_id",
        "pressure_level", "threshold_type",
    ]
    wout = bowl.groupby(wok, dropna=False, as_index=False).agg(
        pressure_deliveries=("source_row_id", "size"),
        valid_bowling_deliveries=("bowling_valid", "sum"),
        legal_balls=("bowling_ball_value", "sum"),
        runs_conceded=("conceded_value", lambda s: s.sum(min_count=1)),
        wickets=("wicket_value", "sum"),
        ambiguous_deliveries=("bowling_unresolved", "sum"),
    )
    wout["economy"] = np.where(
        wout["legal_balls"] > 0, wout["runs_conceded"] * 6 / wout["legal_balls"], np.nan
    )
    wout["bowling_stats_complete"] = wout["ambiguous_deliveries"].eq(0)

    validation = pd.DataFrame([
        ("pressure_state_rows", len(p)),
        ("matched_pressure_deliveries", len(d)),
        ("unmatched_pressure_deliveries", unmatched),
        ("eligible_pressure_deliveries", int(d["pressure_eligible"].fillna(False).sum())),
        ("ambiguous_bowling_deliveries", int(d["bowling_unresolved"].sum())),
        ("batting_output_rows", len(bout)),
        ("bowling_output_rows", len(wout)),
        ("rows_missing_phase", int(d["phase"].isna().sum())),
    ], columns=["check", "value"])

    PROC.mkdir(parents=True, exist_ok=True)
    bout.to_csv(OUT_BAT, index=False)
    wout.to_csv(OUT_BOWL, index=False)
    validation.to_csv(OUT_CHECK, index=False)

    print("Pressure aggregation completed.")
    print(f"Batting: {OUT_BAT} ({len(bout):,} rows)")
    print(f"Bowling: {OUT_BOWL} ({len(wout):,} rows)")
    print(f"Validation: {OUT_CHECK}")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
