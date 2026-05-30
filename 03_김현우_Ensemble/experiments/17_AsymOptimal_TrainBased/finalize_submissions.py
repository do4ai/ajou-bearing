"""Finalize submissions — 1순위 + 백업 2개 명확 명명 + sensitivity heatmap.

챌린지 제출 형식: Bearing, RUL_pred_seconds (필수). 추가 컬럼은 디버그용.

Outputs:
  artifacts/submissions/팀이름_validation_1순위.xlsx
  artifacts/submissions/팀이름_validation_백업1.xlsx
  artifacts/submissions/팀이름_validation_백업2.xlsx
  artifacts/submissions/SUBMISSION_README.md
  artifacts/results/17_AsymOptimal_TrainBased/sensitivity_heatmap.png
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import RESULT_ROOT, ARTIFACT_DIR, add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import VAL_NAMES, asym_score  # noqa: E402

RESULT_DIR = result_dir("17_AsymOptimal_TrainBased")
SUB_DIR = ARTIFACT_DIR / "submissions"
SUB_DIR.mkdir(parents=True, exist_ok=True)


def finalize_submissions() -> None:
    """3개 submission 파일 (1순위 + 백업 2) 표준 포맷으로 저장."""
    print("\n[Finalizing submissions]")

    # 1순위: 18_PerBearing_Robust
    p1 = pd.read_excel(RESULT_DIR / "18_per_bearing_robust_submission.xlsx")
    p1_std = p1[["Bearing", "RUL_pred_seconds"]].copy()
    p1_std["RUL_pred_hours"] = p1_std["RUL_pred_seconds"] / 3600.0
    p1_out = SUB_DIR / "팀이름_validation_1순위.xlsx"
    p1_std.to_excel(p1_out, index=False)
    print(f"  ✅ 1순위 → {p1_out}")
    print(p1_std.to_string(index=False))

    # 백업1: 5_HIBlend_combined (LOBO 검증)
    p2_src = RESULT_ROOT / "05_HIBlend_Baseline_ChannelSym" / "submission_v24_v17v22_combined.xlsx"
    p2 = pd.read_excel(p2_src)
    bear_col = "Bearing" if "Bearing" in p2.columns else "bearing"
    rul_col = next((c for c in ["RUL_pred_seconds", "RUL_blend_combined_s",
                                  "RUL_pred_s"] if c in p2.columns), None)
    p2_std = p2[[bear_col, rul_col]].copy()
    p2_std.columns = ["Bearing", "RUL_pred_seconds"]
    p2_std["RUL_pred_hours"] = p2_std["RUL_pred_seconds"] / 3600.0
    p2_out = SUB_DIR / "팀이름_validation_백업1.xlsx"
    p2_std.to_excel(p2_out, index=False)
    print(f"\n  ✅ 백업1 → {p2_out}")
    print(p2_std.to_string(index=False))

    # 백업2: 19_EOLProgression_Robust
    p3 = pd.read_excel(RESULT_DIR / "19_eol_progression_robust_submission.xlsx")
    p3_std = p3[["Bearing", "RUL_pred_seconds"]].copy()
    p3_std["RUL_pred_hours"] = p3_std["RUL_pred_seconds"] / 3600.0
    p3_out = SUB_DIR / "팀이름_validation_백업2.xlsx"
    p3_std.to_excel(p3_out, index=False)
    print(f"\n  ✅ 백업2 → {p3_out}")
    print(p3_std.to_string(index=False))

    # README
    readme = SUB_DIR / "SUBMISSION_README.md"
    p1_dbg = pd.read_csv(RESULT_DIR / "18_per_bearing_robust_debug.csv")
    sub_rows = [
        "# Submission Files\n",
        "## 1순위 (권장)\n",
        f"- 파일: `팀이름_validation_1순위.xlsx`",
        f"- 전략: per_bearing_best_mix (18_PerBearing_Robust)",
        f"- Sensitivity mean (HI-band prior): **0.4883**",
        f"- 베어링별 best method 선택:\n",
    ]
    for _, r in p1_dbg.iterrows():
        sub_rows.append(f"  - **{r['Bearing']}** (HI={r['HI_last']:.3f}, band={r['band']}): "
                         f"{r['best_method']} → {r['RUL_pred_seconds']:.0f}s "
                         f"(robust={r['robust_mean_score']:.3f})\n")
    sub_rows.extend([
        "\n## 백업 1\n",
        "- 파일: `팀이름_validation_백업1.xlsx`",
        "- 전략: 5_HIBlend_Baseline_ChannelSym combined",
        "- LOBO Combined: **0.712** (검증된 안정 default)",
        "- Sensitivity mean: 0.399\n",
        "\n## 백업 2\n",
        "- 파일: `팀이름_validation_백업2.xlsx`",
        "- 전략: 19_EOLProgression_Robust (Train HI 곡선 fit + EOL bound cap)",
        "- LOBO mean: **0.750** (Train1/2/4 perfect, Train3 fold 약점 공통)",
        "- Sensitivity mean: 0.429\n",
        "\n## 백업 선택 근거: 위험 분산\n",
        "- 1순위 = Sensitivity (HI-band prior)",
        "- 백업1 = LOBO (train 600s last 라벨 fit)",
        "- 백업2 = EOL physics (HI 곡선 → progression)",
        "- 세 가지 다른 가설로 robust 보장.\n",
    ])
    readme.write_text("\n".join(sub_rows))
    print(f"\n  ✅ README → {readme}")


def build_sensitivity_heatmap() -> None:
    """Sensitivity matrix를 heatmap PNG로 시각화 (PPT/Report용)."""
    print("\n[Building sensitivity heatmap]")
    sens = pd.read_csv(RESULT_DIR / "sensitivity_matrix.csv")
    candidates = ["5_HIBlend_combined", "16_balanced", "16_safe", "16_aggressive",
                  "17_asym", "17_hybrid",
                  "19_robust_asym", "19_robust_median",
                  "28_eol_cons", "28_eol_med"]

    # matrix: rows=bearings, cols=candidates, values=sens_mean
    M = []
    for _, row in sens.iterrows():
        M.append([float(row[f"{c}_mean"]) for c in candidates])
    M = np.array(M)
    bearings = sens["bearing"].tolist()

    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(M, cmap="YlGn", vmin=0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(candidates)))
    ax.set_xticklabels(candidates, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(bearings)))
    ax.set_yticklabels(bearings, fontsize=9)
    # Annotate cells
    for i in range(len(bearings)):
        for j in range(len(candidates)):
            color = "white" if M[i, j] > 0.5 else "#1F2A38"
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                     fontsize=7, color=color)
    ax.set_title("Sensitivity expected score per (bearing × candidate)\nHI-band prior, asymmetric penalty grid",
                  fontsize=10, fontweight="bold", color="#051C41")
    fig.colorbar(im, ax=ax, label="Expected asym_score")
    plt.tight_layout()
    out = RESULT_DIR / "sensitivity_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {out}")


def build_per_bearing_chart() -> None:
    """Per-bearing best comparison chart (1순위 vs 백업들)."""
    print("\n[Building per-bearing comparison chart]")
    p1 = pd.read_excel(RESULT_DIR / "18_per_bearing_robust_submission.xlsx")
    p2_src = RESULT_ROOT / "05_HIBlend_Baseline_ChannelSym" / "submission_v24_v17v22_debug.csv"
    p2 = pd.read_csv(p2_src)
    p3 = pd.read_excel(RESULT_DIR / "19_eol_progression_robust_submission.xlsx")

    bearings = p1["Bearing"].tolist()
    p1_vals = p1["RUL_pred_seconds"].values / 3600
    p2_vals = p2["RUL_blend_combined_s"].values / 3600
    p3_vals = p3["RUL_pred_seconds"].values / 3600

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(bearings))
    w = 0.27
    ax.bar(x - w, p1_vals, w, label="1순위: 18_PerBearing_Robust",
            color="#DB9A2D", alpha=0.9)
    ax.bar(x, p2_vals, w, label="백업1: 5_HIBlend (LOBO 0.712)",
            color="#205BA8", alpha=0.9)
    ax.bar(x + w, p3_vals, w, label="백업2: 19_EOLProg_Robust (LOBO 0.75)",
            color="#2D8C5E", alpha=0.9)
    # Value labels
    for i, (v1, v2, v3) in enumerate(zip(p1_vals, p2_vals, p3_vals)):
        ax.text(i - w, v1 + 0.3, f"{v1:.1f}h", ha="center", fontsize=7, color="#1F2A38")
        ax.text(i, v2 + 0.3, f"{v2:.1f}h", ha="center", fontsize=7, color="#1F2A38")
        ax.text(i + w, v3 + 0.3, f"{v3:.1f}h", ha="center", fontsize=7, color="#1F2A38")
    ax.set_xticks(x)
    ax.set_xticklabels(bearings, fontsize=9)
    ax.set_ylabel("Predicted RUL (hours)", fontsize=9)
    ax.set_title("Per-bearing RUL predictions: 1순위 + 백업 2개", fontsize=11,
                  fontweight="bold", color="#051C41")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(p1_vals.max(), p2_vals.max(), p3_vals.max()) * 1.2)
    plt.tight_layout()
    out = RESULT_DIR / "per_bearing_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {out}")


def main() -> None:
    print("=" * 70)
    print("Finalize Submissions + Visualizations")
    print("=" * 70)
    finalize_submissions()
    build_sensitivity_heatmap()
    build_per_bearing_chart()
    print("\n[Done]")


if __name__ == "__main__":
    main()
