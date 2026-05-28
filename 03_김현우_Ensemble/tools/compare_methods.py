"""Compare LOBO scores across named methods and emit summary table.

Reads:
  results/lobo_results.csv  (v17 / 원본 baseline)
  results/lobo_v18.csv
  results/lobo_v22.csv
  results/lobo_v25.csv  (newly trained)
  results/lobo_v26.csv  (newly trained)

Writes:
  results/version_comparison.csv
  results/version_comparison.md
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

R = Path(__file__).resolve().parent / "results"


def safe_load(name, fallback=None):
    p = R / name
    if p.exists():
        return pd.read_csv(p)
    return fallback


def summarize(name, df, full_col_candidates, last_col_candidates):
    if df is None:
        return {"version": name, "full_mean": np.nan, "last_med": np.nan, "combined": np.nan, "note": "missing"}
    full = next((c for c in full_col_candidates if c in df.columns), None)
    last = next((c for c in last_col_candidates if c in df.columns), None)
    if full is None or last is None:
        return {"version": name, "full_mean": np.nan, "last_med": np.nan, "combined": np.nan,
                "note": f"cols: {list(df.columns)[:6]}"}
    fm = df[full].mean(); lm = df[last].mean()
    comb = 0.7 * fm + 0.3 * lm
    return {"version": name, "full_mean": fm, "last_med": lm, "combined": comb, "note": ""}


rows = []
v17 = safe_load("lobo_results.csv")
v18 = safe_load("lobo_v18.csv")
v22 = safe_load("lobo_v22.csv")
v25 = safe_load("lobo_v25.csv")
v26 = safe_load("lobo_v26.csv")

rows.append(summarize("1_Baseline_1ch_TFTBiLSTM_GPR", v17,
                       ["full_med", "full_mean", "score", "asym_score"],
                       ["last_med", "last_mean", "last_score"]))
rows.append(summarize("2_EOLDirect_4ch_WeightedRUL", v18,
                       ["full_med", "full_mean"], ["last_med", "last_mean"]))
rows.append(summarize("4_ChannelSym_EOLWeighted", v22,
                       ["full_med", "full_mean"], ["last_med", "last_mean"]))
rows.append(summarize("6_Dynamics_DTW_TFTBiLSTM", v25,
                       ["full_med", "full_mean"], ["last_med", "last_mean"]))
rows.append(summarize("7_DomainAdv_Dynamics_TFT", v26,
                       ["full_med", "full_mean"], ["last_med", "last_mean"]))

# Static blend numbers (from earlier diagnose)
blend_static = [
    {"version": "3_HIBlend_Baseline_EOLDirect", "full_mean": 0.603, "last_med": 0.978, "combined": 0.715, "note": "from blend_v17v18_grid"},
    {"version": "5_HIBlend_Baseline_ChannelSym combined", "full_mean": 0.599, "last_med": 0.975, "combined": 0.712, "note": "from blend_v17v22_grid (current stable best)"},
    {"version": "5_HIBlend_Baseline_ChannelSym full", "full_mean": 0.637, "last_med": 0.308, "combined": 0.538, "note": "high full but risky last"},
]
rows.extend(blend_static)

cmp_df = pd.DataFrame(rows)
cmp_df["combined"] = cmp_df["combined"].round(4)
cmp_df["full_mean"] = cmp_df["full_mean"].round(4)
cmp_df["last_med"] = cmp_df["last_med"].round(4)

print("=" * 80)
print("Method Comparison (LOBO 4-fold averages)")
print("=" * 80)
print(cmp_df.to_string(index=False))

cmp_df.to_csv(R / "version_comparison.csv", index=False)

md = ["# Method Comparison — LOBO 4-fold", ""]
md.append("> 0.7×Full + 0.3×Last = Combined. 모든 점수는 1.0이 최고.")
md.append("")
md.append("| Method | Full | Last | Combined | Note |")
md.append("|---------|------|------|----------|------|")
for r in rows:
    fm = "—" if pd.isna(r["full_mean"]) else f"{r['full_mean']:.4f}"
    lm = "—" if pd.isna(r["last_med"]) else f"{r['last_med']:.4f}"
    cb = "—" if pd.isna(r["combined"]) else f"{r['combined']:.4f}"
    md.append(f"| {r['version']} | {fm} | {lm} | {cb} | {r.get('note','')} |")
md.append("")
md.append("## 분석")
md.append("- `5_HIBlend_Baseline_ChannelSym` combined가 현재 안정 best 후보.")
md.append("- `6_Dynamics_DTW_TFTBiLSTM`, `7_DomainAdv_Dynamics_TFT`는 Train3 last 폭주 때문에 EOL gate 없이 제출 금지.")
md.append("- 다음 우선순위는 `13_EOLHazardGate_Calibrator`와 `14_RPMAwareOrderFeatures`.")
(R / "version_comparison.md").write_text("\n".join(md))
print(f"\nSaved: {R / 'version_comparison.csv'}")
print(f"Saved: {R / 'version_comparison.md'}")
