#!/usr/bin/env python3
"""Bootstrap validation of pressure-performance reliability.

Resamples each player's observed High-pressure matches and evaluates the
stability of a match-level performance proxy:
  Batting: aggregate High-pressure runs / balls faced (strike rate)
  Bowling: aggregate High-pressure runs conceded / legal balls (economy)

Within each competition/evaluation population, bootstrap percentile scores are
compared before and after the existing candidate reliability shrinkage
(unshrunk/light/medium/strong). This is a diagnostic, NOT a final PPI bootstrap:
it does not resample delivery-level data or all composite-index components.

Run from project root:
    python src/pressure_bootstrap_reliability.py

Inputs:
  outputs/analysis/pressure_index_diagnostics/
      batting_high_pressure_match_diagnostics.csv
      bowling_high_pressure_match_diagnostics.csv
  outputs/analysis/integrated_pressure_diagnostics/
      integrated_batting_pressure_components.csv
      integrated_bowling_pressure_components.csv

Outputs:
  outputs/analysis/pressure_bootstrap_reliability/
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MATCH_DIR = ROOT / "outputs" / "analysis" / "pressure_index_diagnostics"
COMP_DIR = ROOT / "outputs" / "analysis" / "integrated_pressure_diagnostics"
OUT_DIR = ROOT / "outputs" / "analysis" / "pressure_bootstrap_reliability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 500
SEED = 20261115
MIN_MATCHES_FOR_BOOTSTRAP = 2
TOP_N = 10
POP_KEYS = ["competition_group", "evaluation_period"]

SCENARIOS = {
    "unshrunk": (0.0, 0.0),
    "light": (20.0, 2.0),
    "medium": (40.0, 4.0),
    "strong": (70.0, 6.0),
}


def read_csv(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    d = pd.read_csv(path)
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    return d


def percentile_score(values: pd.Series, higher_better: bool) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    if not higher_better:
        x = -x
    return x.rank(method="average", pct=True) * 100.0


def reliability(volume: float, matches: float, volume_k: float, matches_k: float) -> float:
    if not np.isfinite(volume) or not np.isfinite(matches):
        return np.nan
    if volume_k == 0 and matches_k == 0:
        return 1.0
    rv = volume / (volume + volume_k) if volume + volume_k > 0 else 0.0
    rm = matches / (matches + matches_k) if matches + matches_k > 0 else 0.0
    return float(np.sqrt(rv * rm))


def spearman_safe(a: pd.Series, b: pd.Series) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))


def prepare_role(role: str) -> tuple[pd.DataFrame, pd.DataFrame, str, str, str, bool]:
    if role == "Batting":
        matches = read_csv(
            MATCH_DIR / "batting_high_pressure_match_diagnostics.csv",
            POP_KEYS + ["striker_id", "striker_name", "match_id",
                        "match_high_balls", "match_high_runs"]
        )
        comp = read_csv(
            COMP_DIR / "integrated_batting_pressure_components.csv",
            POP_KEYS + ["striker_id", "striker_name", "high_pressure_balls_support",
                        "high_pressure_matches"]
        )
        id_col, name_col = "striker_id", "striker_name"
        vol_col, n_col = "high_pressure_balls_support", "high_pressure_matches"
        matches["_numerator"] = pd.to_numeric(matches["match_high_runs"], errors="coerce")
        matches["_denominator"] = pd.to_numeric(matches["match_high_balls"], errors="coerce")
        higher_better = True
    else:
        matches = read_csv(
            MATCH_DIR / "bowling_high_pressure_match_diagnostics.csv",
            POP_KEYS + ["bowler_id", "bowler_name", "match_id",
                        "match_high_legal_balls", "match_high_runs_conceded"]
        )
        comp = read_csv(
            COMP_DIR / "integrated_bowling_pressure_components.csv",
            POP_KEYS + ["bowler_id", "bowler_name", "high_legal_balls",
                        "high_pressure_matches"]
        )
        id_col, name_col = "bowler_id", "bowler_name"
        vol_col, n_col = "high_legal_balls", "high_pressure_matches"
        matches["_numerator"] = pd.to_numeric(matches["match_high_runs_conceded"], errors="coerce")
        matches["_denominator"] = pd.to_numeric(matches["match_high_legal_balls"], errors="coerce")
        higher_better = False

    matches["_denominator"] = pd.to_numeric(matches["_denominator"], errors="coerce")
    matches = matches[
        matches["_numerator"].notna()
        & matches["_denominator"].gt(0)
        & matches["match_id"].notna()
    ].copy()

    keys = POP_KEYS + [id_col]
    if matches.duplicated(keys + ["match_id"]).any():
        raise ValueError(f"{role}: duplicate player-match observations detected")
    if comp.duplicated(keys).any():
        raise ValueError(f"{role}: duplicate player-population rows detected")

    keep = keys + [name_col, vol_col, n_col]
    base = comp[keep].copy()
    base[vol_col] = pd.to_numeric(base[vol_col], errors="coerce")
    base[n_col] = pd.to_numeric(base[n_col], errors="coerce")
    return base, matches, id_col, name_col, vol_col, higher_better


def run_role(role: str, rng: np.random.Generator):
    base, match_df, id_col, name_col, volume_col, higher_better = prepare_role(role)
    rows_player, rows_summary, rows_boot = [], [], []
    rank_corr_rows = []

    for pop_key, pop_base in base.groupby(POP_KEYS, dropna=False, sort=True):
        if not isinstance(pop_key, tuple):
            pop_key = (pop_key,)
        pop_match = match_df
        for key, value in zip(POP_KEYS, pop_key):
            pop_match = pop_match[pop_match[key].eq(value)]
        pop_base = pop_base.reset_index(drop=True)
        n_players = len(pop_base)
        groups = {
            pid: (
                g["_numerator"].to_numpy(dtype=float),
                g["_denominator"].to_numpy(dtype=float),
            )
            for pid, g in pop_match.groupby(id_col, sort=False)
        }

        # Match-level resampling is meaningful only with >=2 observed matches.
        usable = pop_base[pop_base[id_col].isin(
            [pid for pid, g in groups.items() if len(g[0]) >= MIN_MATCHES_FOR_BOOTSTRAP]
        )].copy()
        usable_ids = set(usable[id_col].tolist())
        if len(usable) < 2:
            continue

        ids = usable[id_col].tolist()
        # Reliability is calculated from full observed exposure, not the bootstrap draw.
        rel_by_scenario = {}
        for scenario, (vk, mk) in SCENARIOS.items():
            rel_by_scenario[scenario] = {
                row[id_col]: reliability(row[volume_col], row["high_pressure_matches"], vk, mk)
                for _, row in usable.iterrows()
            }

        boot_raw = {pid: [] for pid in ids}
        boot_adj = {s: {pid: [] for pid in ids} for s in SCENARIOS}
        boot_ranks = {s: {pid: [] for pid in ids} for s in SCENARIOS}
        boot_top = {s: {pid: 0 for pid in ids} for s in SCENARIOS}
        concordance = {s: [] for s in SCENARIOS if s != "unshrunk"}

        for b in range(N_BOOT):
            performance = {}
            for pid in ids:
                numer, denom = groups[pid]
                take = rng.integers(0, len(numer), size=len(numer))
                den = denom[take].sum()
                if den > 0:
                    scale = 100.0 if role == "Batting" else 6.0
                    performance[pid] = scale * numer[take].sum() / den
                else:
                    performance[pid] = np.nan

            raw = pd.Series(performance, dtype=float)
            raw_pct = percentile_score(raw, higher_better)
            adjusted = {}
            for scenario in SCENARIOS:
                rel = pd.Series(rel_by_scenario[scenario])
                adj = 50.0 + rel * (raw_pct - 50.0)
                adjusted[scenario] = adj

            for pid in ids:
                boot_raw[pid].append(raw_pct.get(pid, np.nan))
                for scenario in SCENARIOS:
                    boot_adj[scenario][pid].append(adjusted[scenario].get(pid, np.nan))

            rank_series = {}
            for scenario in SCENARIOS:
                ranks = adjusted[scenario].rank(ascending=False, method="average")
                rank_series[scenario] = ranks
                for pid in ids:
                    r = ranks.get(pid, np.nan)
                    boot_ranks[scenario][pid].append(r)
                    if pd.notna(r) and r <= TOP_N:
                        boot_top[scenario][pid] += 1
            for scenario in concordance:
                concordance[scenario].append(
                    spearman_safe(rank_series["unshrunk"], rank_series[scenario])
                )

        # Compare per-player rank dispersion and top-10 frequency by shrinkage.
        scenario_metrics = {}
        for scenario in SCENARIOS:
            rank_sds, top_freqs, score_sds = [], [], []
            for pid in ids:
                rr = pd.Series(boot_ranks[scenario][pid], dtype=float).dropna()
                ss = pd.Series(boot_adj[scenario][pid], dtype=float).dropna()
                rank_sd = rr.std(ddof=1) if len(rr) > 1 else np.nan
                top_freq = boot_top[scenario][pid] / N_BOOT
                score_sd = ss.std(ddof=1) if len(ss) > 1 else np.nan
                rank_sds.append(rank_sd)
                top_freqs.append(top_freq)
                score_sds.append(score_sd)
                rows_player.append({
                    "role": role, **dict(zip(POP_KEYS, pop_key)),
                    id_col: pid,
                    name_col: usable.loc[usable[id_col].eq(pid), name_col].iloc[0],
                    "observed_high_pressure_matches": len(groups[pid][0]),
                    "bootstrap_evaluable": True,
                    "shrinkage_scenario": scenario,
                    "mean_bootstrap_rank": rr.mean(),
                    "bootstrap_rank_sd": rank_sd,
                    "rank_5th_percentile": rr.quantile(0.05),
                    "rank_95th_percentile": rr.quantile(0.95),
                    "top10_selection_frequency": top_freq,
                    "bootstrap_adjusted_score_sd": score_sd,
                    "reliability_factor": rel_by_scenario[scenario][pid],
                })
            scenario_metrics[scenario] = {
                "mean_player_rank_sd": float(np.nanmean(rank_sds)),
                "median_player_rank_sd": float(np.nanmedian(rank_sds)),
                "mean_top10_frequency": float(np.nanmean(top_freqs)),
                "mean_adjusted_score_sd": float(np.nanmean(score_sds)),
            }

        # Per-population summary and paired rank correlation vs unshrunk.
        for scenario, metrics in scenario_metrics.items():
            rows_summary.append({
                "role": role, **dict(zip(POP_KEYS, pop_key)),
                "eligible_player_units": n_players,
                "players_with_match_data": int(pop_base[id_col].isin(groups.keys()).sum()),
                "players_bootstrap_evaluable_2plus_matches": len(usable),
                "players_excluded_single_match_or_no_match": n_players - len(usable),
                "bootstrap_replicates": N_BOOT,
                "shrinkage_scenario": scenario,
                **metrics,
            })
            if scenario != "unshrunk":
                vals = pd.Series(concordance[scenario], dtype=float).dropna()
                rank_corr_rows.append({
                    "role": role, **dict(zip(POP_KEYS, pop_key)),
                    "comparison": f"unshrunk_vs_{scenario}",
                    "mean_bootstrap_cross_sectional_rank_correlation": vals.mean() if len(vals) else np.nan,
                    "median_bootstrap_cross_sectional_rank_correlation": vals.median() if len(vals) else np.nan,
                    "valid_bootstrap_rank_correlations": len(vals),
                })

    return pd.DataFrame(rows_player), pd.DataFrame(rows_summary), pd.DataFrame(rank_corr_rows)


def main():
    rng = np.random.default_rng(SEED)
    all_players, all_summary, all_corr = [], [], []
    for role in ("Batting", "Bowling"):
        p, s, c = run_role(role, rng)
        all_players.append(p)
        all_summary.append(s)
        all_corr.append(c)

    player = pd.concat(all_players, ignore_index=True) if all_players else pd.DataFrame()
    summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    corr = pd.concat(all_corr, ignore_index=True) if all_corr else pd.DataFrame()

    validation = pd.DataFrame([
        {"check": "bootstrap_replicates_per_population_scenario", "value": N_BOOT},
        {"check": "minimum_matches_for_bootstrap_evaluation", "value": MIN_MATCHES_FOR_BOOTSTRAP},
        {"check": "player_scenario_rows", "value": len(player)},
        {"check": "population_scenario_summary_rows", "value": len(summary)},
        {"check": "rank_correlation_rows", "value": len(corr)},
        {"check": "duplicate_player_scenario_keys", "value": int(player.duplicated(["role"] + POP_KEYS + ["striker_id" if "striker_id" in player else "bowler_id", "shrinkage_scenario"]).sum()) if not player.empty else 0},
    ])

    player.to_csv(OUT_DIR / "bootstrap_player_rank_stability.csv", index=False)
    summary.to_csv(OUT_DIR / "bootstrap_population_stability_summary.csv", index=False)
    corr.to_csv(OUT_DIR / "bootstrap_rank_correlation_summary.csv", index=False)
    validation.to_csv(OUT_DIR / "bootstrap_validation.csv", index=False)

    print("PRESSURE BOOTSTRAP RELIABILITY DIAGNOSTICS COMPLETE")
    print(f"Output folder: {OUT_DIR}")
    print("\nPOPULATION STABILITY SUMMARY")
    print(summary.to_string(index=False) if not summary.empty else "No evaluable populations")
    print("\nVALIDATION")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
