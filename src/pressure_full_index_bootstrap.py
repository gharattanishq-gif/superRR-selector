#!/usr/bin/env python3
"""Full composite pressure-index match-cluster bootstrap.

Run from SupeRR project root:
    python src/pressure_full_index_bootstrap.py

Resamples each player's complete observed match clusters with replacement,
keeping High/Moderate/Low pressure rows from a sampled match together. On each
replicate it recalculates absolute performance, High-vs-baseline uplift,
match-to-match consistency, population percentiles and the 70/20/10 composite.
It compares unshrunk/light/medium/strong component-level reliability adjustments.

This is a sensitivity diagnostic, not an automatic choice of weights or
shrinkage. The bootstrap population is the player's observed pressure-match
sample; it cannot create evidence for matches/periods with no pressure records.
"""
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "outputs" / "processed"
ANALYSIS = ROOT / "outputs" / "analysis"
INDEX_DIR = ANALYSIS / "pressure_index_diagnostics"
INTEGRATED_DIR = ANALYSIS / "integrated_pressure_diagnostics"
OUT = ANALYSIS / "pressure_full_index_bootstrap"
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 20
SEED = 20261115
MIN_HIGH_MATCHES = 2
WEIGHTS = (0.70, 0.20, 0.10)
SHRINK = {"unshrunk": (0.0, 0.0), "light": (20.0, 2.0),
          "medium": (40.0, 4.0), "strong": (70.0, 6.0)}
POP_KEYS = ["competition_group", "evaluation_period"]


def require(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_inputs():
    paths = {
        "bat": PROCESSED / "player_match_pressure_batting.csv",
        "bowl": PROCESSED / "player_match_pressure_bowling.csv",
        "dots": INDEX_DIR / "player_match_pressure_bowling_dotballs.csv",
        "eb": INTEGRATED_DIR / "integrated_batting_pressure_components.csv",
        "ew": INTEGRATED_DIR / "integrated_bowling_pressure_components.csv",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))
    bat, bowl, dots, eb, ew = [pd.read_csv(paths[k]) for k in ["bat", "bowl", "dots", "eb", "ew"]]
    require(bat, ["match_id", "competition_group", "season", "innings", "phase", "striker_id", "striker_name", "pressure_level", "balls_faced", "batter_runs", "fours", "sixes", "dismissals"], "batting pressure")
    require(bowl, ["match_id", "competition_group", "season", "innings", "phase", "bowler_id", "bowler_name", "pressure_level", "legal_balls", "runs_conceded", "wickets"], "bowling pressure")
    require(dots, ["match_id", "competition_group", "season", "innings", "phase", "bowler_id", "pressure_level", "dot_balls"], "dot-ball pressure")
    require(eb, ["competition_group", "evaluation_period", "striker_id", "striker_name", "eligible_by_matches"], "eligible batters")
    require(ew, ["competition_group", "evaluation_period", "bowler_id", "bowler_name", "eligible_by_matches"], "eligible bowlers")
    for d in (bat, bowl, dots):
        d["evaluation_period"] = np.where(d["competition_group"].eq("IPL"), d["season"].astype(str), "2022-2026 combined")
        d["evaluation_period"] = d["evaluation_period"].astype(str)
        d["match_id"] = pd.to_numeric(d["match_id"], errors="coerce")
        d["innings"] = pd.to_numeric(d["innings"], errors="coerce")
    num(bat, ["balls_faced", "batter_runs", "fours", "sixes", "dismissals"])
    num(bowl, ["legal_balls", "runs_conceded", "wickets"])
    num(dots, ["dot_balls"])
    dot_key = ["match_id", "competition_group", "evaluation_period", "innings", "phase", "bowler_id", "pressure_level"]
    if dots.duplicated(dot_key).any():
        raise ValueError("Duplicate keys found in bowling dot-ball file.")
    if bowl.duplicated(dot_key).any():
        raise ValueError("Duplicate keys found in bowling pressure file.")
    bowl = bowl.merge(dots[dot_key + ["dot_balls"]], on=dot_key, how="left", validate="one_to_one")
    if bowl.dot_balls.isna().any():
        raise ValueError(f"{int(bowl.dot_balls.isna().sum())} bowling pressure rows lack dot-ball matches.")
    eb = eb.loc[eb.eligible_by_matches.fillna(False)].copy()
    ew = ew.loc[ew.eligible_by_matches.fillna(False)].copy()
    return bat, bowl, eb, ew


def prep_population(data, eligible, id_col, role):
    eligible = eligible.copy()
    eligible[id_col] = pd.to_numeric(eligible[id_col], errors="coerce")
    data[id_col] = pd.to_numeric(data[id_col], errors="coerce")
    data = data[data[id_col].isin(set(eligible[id_col].dropna()))].copy()
    out = {}
    excluded = 0
    for pid, g in data.groupby(id_col, sort=False):
        high_matches = g.loc[g.pressure_level.eq("High"), "match_id"].nunique()
        match_ids = g.match_id.dropna().unique()
        if high_matches < MIN_HIGH_MATCHES or len(match_ids) < MIN_HIGH_MATCHES:
            excluded += 1
            continue
        out[int(pid)] = (g, match_ids)
    return data, out, excluded


def make_sample_draws(player_map, rng):
    parts = []
    for pid, (g, match_ids) in player_map.items():
        chosen = rng.choice(match_ids, size=len(match_ids), replace=True)
        draws = pd.DataFrame({"_draw_id": np.arange(len(chosen)), "match_id": chosen})
        draws["_player_id"] = pid
        parts.append(draws)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def sampled_rows(data, draws, id_col):
    merged = data.merge(draws, left_on=[id_col, "match_id"], right_on=["_player_id", "match_id"], how="inner", validate="many_to_many")
    return merged.drop(columns=[id_col])


def pivot_pressure(agg, id_col, values):
    p = agg.pivot_table(index=id_col, columns="pressure_level", values=values, aggfunc="sum")
    p.columns = [f"{v}_{lev}" for v, lev in p.columns]
    return p


def batting_metrics(s):
    pid = "_player_id"
    a = s.groupby([pid, "pressure_level"], dropna=False).agg(
        balls=("balls_faced", "sum"), runs=("batter_runs", "sum"),
        outs=("dismissals", "sum"), fours=("fours", "sum"), sixes=("sixes", "sum")
    ).reset_index()
    p = pivot_pressure(a, pid, ["balls", "runs", "outs", "fours", "sixes"])
    for c in ["balls_High", "runs_High", "outs_High", "fours_High", "sixes_High",
              "balls_Low", "balls_Moderate", "runs_Low", "runs_Moderate", "outs_Low", "outs_Moderate"]:
        if c not in p: p[c] = np.nan
    p["base_balls"] = p[["balls_Low", "balls_Moderate"]].fillna(0).sum(axis=1)
    p["base_runs"] = p[["runs_Low", "runs_Moderate"]].fillna(0).sum(axis=1)
    p["base_outs"] = p[["outs_Low", "outs_Moderate"]].fillna(0).sum(axis=1)
    p["abs_sr"] = 100 * p.runs_High / p.balls_High.where(p.balls_High.gt(0))
    p["abs_average"] = p.runs_High / p.outs_High.where(p.outs_High.gt(0))
    p["abs_boundary"] = 100 * (4*p.fours_High + 6*p.sixes_High) / p.runs_High.where(p.runs_High.gt(0))
    p["base_sr"] = 100 * p.base_runs / p.base_balls.where(p.base_balls.gt(0))
    p["base_average"] = p.base_runs / p.base_outs.where(p.base_outs.gt(0))
    # Comparable innings-phase SR uplift, weighted by High-pressure balls.
    st = s.groupby([pid, "innings", "phase", "pressure_level"], dropna=False).agg(
        balls=("balls_faced", "sum"), runs=("batter_runs", "sum")
    ).reset_index()
    q = st.pivot_table(index=[pid, "innings", "phase"], columns="pressure_level", values=["balls", "runs"], aggfunc="sum")
    q.columns = [f"{v}_{lev}" for v, lev in q.columns]
    for c in ["balls_High", "runs_High", "balls_Low", "balls_Moderate", "runs_Low", "runs_Moderate"]:
        if c not in q: q[c] = 0.0
    q["base_balls"] = q[["balls_Low", "balls_Moderate"]].fillna(0).sum(axis=1)
    q["base_runs"] = q[["runs_Low", "runs_Moderate"]].fillna(0).sum(axis=1)
    q["uplift"] = 100*q.runs_High/q.balls_High.where(q.balls_High.gt(0)) - 100*q.base_runs/q.base_balls.where(q.base_balls.gt(0))
    q["weight"] = q.balls_High.where(q.uplift.notna(), 0)
    us = (q.assign(wx=q.uplift*q.weight).groupby(level=0).agg(wx=("wx", "sum"), w=("weight", "sum")))
    p["uplift_sr"] = us.wx / us.w.where(us.w.gt(0))
    p["uplift_average"] = p.abs_average - p.base_average
    hm = s[s.pressure_level.eq("High")].groupby([pid, "_draw_id"]).agg(runs=("batter_runs", "sum"), balls=("balls_faced", "sum"))
    hm = hm[hm.balls.gt(0)].copy()
    hm["match_sr"] = 100*hm.runs/hm.balls
    cons = hm.groupby(level=0).match_sr.apply(lambda x: (x-x.median()).abs().median() if len(x)>=2 else np.nan)
    p["consistency"] = cons
    match_counts = hm.groupby(level=0).size()
    p["high_matches"] = match_counts
    p["high_balls"] = p.balls_High
    p["high_dismissals"] = p.outs_High
    p["matched_balls"] = np.minimum(p.balls_High, p.base_balls)
    p["matched_dismissals"] = np.minimum(p.outs_High, p.base_outs)
    return p.reset_index().rename(columns={pid:"player_id"})


def bowling_metrics(s):
    pid = "_player_id"
    a = s.groupby([pid, "pressure_level"], dropna=False).agg(
        balls=("legal_balls", "sum"), runs=("runs_conceded", "sum"),
        wickets=("wickets", "sum"), dots=("dot_balls", "sum")
    ).reset_index()
    p = pivot_pressure(a, pid, ["balls", "runs", "wickets", "dots"])
    for c in ["balls_High", "runs_High", "wickets_High", "dots_High", "balls_Low", "balls_Moderate", "runs_Low", "runs_Moderate", "wickets_Low", "wickets_Moderate"]:
        if c not in p: p[c] = np.nan
    p["base_balls"] = p[["balls_Low", "balls_Moderate"]].fillna(0).sum(axis=1)
    p["base_runs"] = p[["runs_Low", "runs_Moderate"]].fillna(0).sum(axis=1)
    p["base_wickets"] = p[["wickets_Low", "wickets_Moderate"]].fillna(0).sum(axis=1)
    p["abs_economy"] = 6*p.runs_High/p.balls_High.where(p.balls_High.gt(0))
    p["abs_wicket_rate"] = 6*p.wickets_High/p.balls_High.where(p.balls_High.gt(0))
    p["abs_dot_pct"] = 100*p.dots_High/p.balls_High.where(p.balls_High.gt(0))
    p["base_economy"] = 6*p.base_runs/p.base_balls.where(p.base_balls.gt(0))
    p["base_wicket_rate"] = 6*p.base_wickets/p.base_balls.where(p.base_balls.gt(0))
    p["uplift_economy"] = p.base_economy-p.abs_economy
    p["uplift_wicket_rate"] = p.abs_wicket_rate-p.base_wicket_rate
    # High-vs-Low/Moderate dot-ball uplift within innings-phase, weighted by High legal balls.
    st = s.groupby([pid, "innings", "phase", "pressure_level"], dropna=False).agg(
        balls=("legal_balls", "sum"), dots=("dot_balls", "sum")
    ).reset_index()
    q = st.pivot_table(index=[pid, "innings", "phase"], columns="pressure_level", values=["balls", "dots"], aggfunc="sum")
    q.columns = [f"{v}_{lev}" for v, lev in q.columns]
    for c in ["balls_High", "dots_High", "balls_Low", "balls_Moderate", "dots_Low", "dots_Moderate"]:
        if c not in q: q[c] = 0.0
    q["base_balls"] = q[["balls_Low", "balls_Moderate"]].fillna(0).sum(axis=1)
    q["base_dots"] = q[["dots_Low", "dots_Moderate"]].fillna(0).sum(axis=1)
    q["uplift"] = 100*q.dots_High/q.balls_High.where(q.balls_High.gt(0)) - 100*q.base_dots/q.base_balls.where(q.base_balls.gt(0))
    q["weight"] = q.balls_High.where(q.uplift.notna(), 0)
    us = q.assign(wx=q.uplift*q.weight).groupby(level=0).agg(wx=("wx", "sum"), w=("weight", "sum"))
    p["uplift_dot"] = us.wx/us.w.where(us.w.gt(0))
    p["dot_uplift_strata"] = q.groupby(level=0).uplift.count()
    hm = s[s.pressure_level.eq("High")].groupby([pid, "_draw_id"]).agg(runs=("runs_conceded", "sum"), balls=("legal_balls", "sum"))
    hm = hm[hm.balls.gt(0)].copy()
    hm["match_economy"] = 6*hm.runs/hm.balls
    cons = hm.groupby(level=0).match_economy.apply(lambda x: (x-x.median()).abs().median() if len(x)>=2 else np.nan)
    p["consistency"] = cons
    p["high_matches"] = hm.groupby(level=0).size()
    p["high_balls"] = p.balls_High
    p["matched_balls"] = np.minimum(p.balls_High, p.base_balls)
    return p.reset_index().rename(columns={pid:"player_id"})


def score_replicate(metrics, role, comp, period, rep):
    d = metrics.copy()
    if role == "Batting":
        directions = {"abs_sr":True,"abs_average":True,"abs_boundary":True,"uplift_sr":True,"uplift_average":True,"consistency":False}
        blocks = {"absolute":["abs_sr","abs_average","abs_boundary"],"uplift":["uplift_sr","uplift_average"],"consistency":["consistency"]}
    else:
        directions = {"abs_economy":False,"abs_wicket_rate":True,"abs_dot_pct":True,"uplift_economy":True,"uplift_wicket_rate":True,"uplift_dot":True,"consistency":False}
        blocks = {"absolute":["abs_economy","abs_wicket_rate","abs_dot_pct"],"uplift":["uplift_economy","uplift_wicket_rate","uplift_dot"],"consistency":["consistency"]}
    for m, high in directions.items():
        d[m+"_pct"] = d[m].rank(method="average", pct=True, ascending=high)*100
    out=[]
    for scenario,(vk,mk) in SHRINK.items():
        x=d.copy()
        for m in directions:
            if m=="consistency":
                rel=np.sqrt(1.0*(x.high_matches/(x.high_matches+mk))) if mk else pd.Series(1.0,index=x.index)
            else:
                if m in ("abs_average",): vol=x.high_dismissals; vkk={"light":2,"medium":4,"strong":6,"unshrunk":0}[scenario]
                elif m=="uplift_average": vol=x.matched_dismissals; vkk={"light":2,"medium":4,"strong":6,"unshrunk":0}[scenario]
                elif m.startswith("uplift_"): vol=x.matched_balls; vkk=vk
                else: vol=x.high_balls; vkk=vk
                rv=pd.Series(1.0,index=x.index) if vkk==0 else vol.clip(lower=0)/(vol.clip(lower=0)+vkk)
                rm=pd.Series(1.0,index=x.index) if mk==0 else x.high_matches/(x.high_matches+mk)
                rel=np.sqrt(rv*rm)
            x[m+"_adj"]=50+rel*(x[m+"_pct"]-50)
            if m=="uplift_dot":
                x.loc[x.dot_uplift_strata.fillna(0).lt(1),m+"_adj"]=np.nan
        for block, ms in blocks.items():
            cols=[m+"_adj" for m in ms]
            x[block+"_score"]=x[cols].mean(axis=1).where(x[cols].notna().all(axis=1))
        x["ppi_score"]=WEIGHTS[0]*x.absolute_score+WEIGHTS[1]*x.uplift_score+WEIGHTS[2]*x.consistency_score
        x["rank"]=x.ppi_score.rank(method="min",ascending=False)
        x["top10"]=x["rank"].le(10)
        x["role"]=role; x["competition_group"]=comp; x["evaluation_period"]=str(period)
        x["bootstrap_rep"]=rep; x["shrinkage_scenario"]=scenario
        out.append(x[["role","competition_group","evaluation_period","player_id","bootstrap_rep","shrinkage_scenario","ppi_score","rank","top10"]])
    return pd.concat(out,ignore_index=True)


def run_role(role,data,eligible,id_col,metric_fn,rng):
    all_rows=[]; val=[]
    for (comp,period), ep in eligible.groupby(POP_KEYS,dropna=False):
        d=data[(data.competition_group==comp)&(data.evaluation_period.astype(str)==str(period))].copy()
        d, player_map, excluded=prep_population(d,ep,id_col,role)
        if len(player_map)<10:
            val.append({"role":role,"competition_group":comp,"evaluation_period":period,"eligible_units":len(ep),"bootstrap_evaluable_players":len(player_map),"excluded_single_or_no_high_match":excluded,"status":"insufficient"}); continue
        for rep in range(N_BOOT):
            draws=make_sample_draws(player_map,rng)
            s=sampled_rows(d,draws,id_col)
            metrics=metric_fn(s)
            metrics=metrics.merge(ep[[id_col]].rename(columns={id_col:"player_id"}),left_on="player_id",right_on="player_id",how="inner")
            scored=score_replicate(metrics,role,comp,period,rep)
            all_rows.append(scored)
        val.append({"role":role,"competition_group":comp,"evaluation_period":period,"eligible_units":len(ep),"players_with_pressure_rows":d[id_col].nunique(),"bootstrap_evaluable_players":len(player_map),"excluded_single_or_no_high_match":excluded,"bootstrap_repetitions":N_BOOT,"status":"ok"})
    return pd.concat(all_rows,ignore_index=True) if all_rows else pd.DataFrame(),pd.DataFrame(val)


def main():
    warnings.filterwarnings("ignore",category=RuntimeWarning)
    bat,bowl,eb,ew=load_inputs()
    rng=np.random.default_rng(SEED)
    bat_scores,bat_val=run_role("Batting",bat,eb,"striker_id",batting_metrics,rng)
    bowl_scores,bowl_val=run_role("Bowling",bowl,ew,"bowler_id",bowling_metrics,rng)
    scores=pd.concat([bat_scores,bowl_scores],ignore_index=True)
    validation=pd.concat([bat_val,bowl_val],ignore_index=True)
    if scores.empty: raise RuntimeError("No bootstrap score rows generated.")
    player=scores.groupby(["role","competition_group","evaluation_period","player_id","shrinkage_scenario"],dropna=False).agg(
        bootstrap_repetitions=("bootstrap_rep","nunique"),mean_rank=("rank","mean"),median_rank=("rank","median"),rank_sd=("rank","std"),
        rank_p025=("rank",lambda x:x.quantile(.025)),rank_p975=("rank",lambda x:x.quantile(.975)),mean_score=("ppi_score","mean"),score_sd=("ppi_score","std"),top10_frequency=("top10","mean")
    ).reset_index()
    # Population stability is computed from each player's rank SD across bootstrap
    # repetitions (not the cross-sectional SD of ranks within one replicate).
    pop=player.groupby(["role","competition_group","evaluation_period","shrinkage_scenario"],dropna=False).agg(
        players=("player_id","nunique"),
        mean_player_rank_sd=("rank_sd","mean"),
        median_player_rank_sd=("rank_sd","median"),
        p90_player_rank_sd=("rank_sd",lambda x:x.quantile(.90)),
        mean_top10_frequency=("top10_frequency","mean"),
        p90_top10_frequency=("top10_frequency",lambda x:x.quantile(.90)),
        players_top10_frequency_ge_50pct=("top10_frequency",lambda x:int((x>=.50).sum())),
        mean_score_sd=("score_sd","mean")
    ).reset_index()
    comparisons=[]
    for (role,comp,period,rep),g in scores.groupby(["role","competition_group","evaluation_period","bootstrap_rep"],dropna=False):
        tops={sc:set(x.loc[x.top10,"player_id"]) for sc,x in g.groupby("shrinkage_scenario")}
        for a,b in [("unshrunk","light"),("unshrunk","medium"),("unshrunk","strong"),("light","medium"),("light","strong"),("medium","strong")]:
            if a in tops and b in tops:
                union=tops[a]|tops[b]
                comparisons.append({"role":role,"competition_group":comp,"evaluation_period":period,"bootstrap_rep":rep,"scenario_a":a,"scenario_b":b,
                                    "top10_jaccard":len(tops[a]&tops[b])/len(union) if union else np.nan,"common_top10":len(tops[a]&tops[b])})
    comp=pd.DataFrame(comparisons)
    comp_summary=comp.groupby(["role","competition_group","evaluation_period","scenario_a","scenario_b"],dropna=False).agg(
        repetitions=("bootstrap_rep","nunique"),mean_top10_jaccard=("top10_jaccard","mean"),median_top10_jaccard=("top10_jaccard","median"),mean_common_top10=("common_top10","mean")
    ).reset_index() if not comp.empty else pd.DataFrame()
    checks=pd.DataFrame([
        ("batting_duplicate_source_keys",int(bat.duplicated(["match_id","competition_group","evaluation_period","innings","phase","striker_id","pressure_level"]).sum())),
        ("bowling_duplicate_source_keys",int(bowl.duplicated(["match_id","competition_group","evaluation_period","innings","phase","bowler_id","pressure_level"]).sum())),
        ("bowling_rows_without_dot_ball_match",int(bowl.dot_balls.isna().sum())),
        ("bootstrap_repetitions",N_BOOT),("minimum_high_pressure_matches",MIN_HIGH_MATCHES),
        ("shrinkage_scenarios",len(SHRINK)),("composite_weights","70/20/10"),
        ("scores_outside_0_100",int(((scores.ppi_score<0)|(scores.ppi_score>100)).sum())),
    ],columns=["check","value"])
    scores.to_csv(OUT/"full_index_bootstrap_replicate_scores.csv",index=False)
    player.to_csv(OUT/"full_index_bootstrap_player_stability.csv",index=False)
    pop.to_csv(OUT/"full_index_bootstrap_population_summary.csv",index=False)
    comp.to_csv(OUT/"full_index_bootstrap_top10_comparisons.csv",index=False)
    comp_summary.to_csv(OUT/"full_index_bootstrap_top10_summary.csv",index=False)
    validation.to_csv(OUT/"full_index_bootstrap_population_validation.csv",index=False)
    checks.to_csv(OUT/"full_index_bootstrap_validation.csv",index=False)
    print("FULL COMPOSITE PRESSURE-INDEX BOOTSTRAP COMPLETE")
    print(f"Output folder: {OUT}")
    print(f"Bootstrap repetitions: {N_BOOT}")
    print("\nPOPULATION VALIDATION\n",validation.to_string(index=False))
    print("\nPOPULATION STABILITY SUMMARY\n",pop.to_string(index=False))
    print("\nVALIDATION CHECKS\n",checks.to_string(index=False))
    print("\nNo final weights or shrinkage parameters have been selected.")

if __name__=="__main__": main()
