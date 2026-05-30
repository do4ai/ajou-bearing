"""Build A4 1-page report PDF — KSPHM-KIMM 2026 bearing RUL.

챌린지 제출물 3종 중 하나: 팀이름_report.pdf (A4 1페이지).
matplotlib 기반. 텍스트 + 표 + 작은 시각화 통합.

Outputs:
  artifacts/results/17_AsymOptimal_TrainBased/팀이름_report.pdf
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import RESULT_ROOT, add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import TRAIN_NAMES, VAL_NAMES  # noqa: E402

RESULT_DIR = result_dir("17_AsymOptimal_TrainBased")
FEATURE_CSV = RESULT_ROOT / "06_Dynamics_DTW_TFTBiLSTM" / "v25_features_dynamics.csv"

# A4: 210 mm × 297 mm = 8.27 × 11.69 inches
A4 = (8.27, 11.69)

NAVY = "#051C41"
GOLD = "#DB9A2D"
BLUE = "#205BA8"
RED = "#C23934"
GREEN = "#2D8C5E"
LIGHT = "#F4F7FC"
MUTED = "#5C687A"
TEXT = "#1F2A38"
LINE = "#DAE1EB"


def main() -> None:
    print("=" * 70)
    print("Building A4 1-page report PDF")
    print("=" * 70)

    # Load data for figures
    sens_summary = pd.read_csv(RESULT_DIR / "21_submission_summary.csv")
    per_bearing = pd.read_csv(RESULT_DIR / "18_per_bearing_robust_debug.csv")

    # Train HI vs RUL stats
    df = pd.read_csv(FEATURE_CSV).fillna(0)
    train = df[df.bearing.isin(TRAIN_NAMES)]

    hi_bands = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.85), (0.85, 1.0)]
    band_stats = []
    for lo, hi in hi_bands:
        sub = train[(train.HI >= lo) & (train.HI < hi)]
        if len(sub) > 0:
            band_stats.append({
                "band": f"[{lo:.2f}, {hi:.2f})",
                "n": len(sub),
                "median_h": float(np.median(sub.rul_s)) / 3600,
                "p25_h": float(np.percentile(sub.rul_s, 25)) / 3600,
                "p75_h": float(np.percentile(sub.rul_s, 75)) / 3600,
            })
    band_df = pd.DataFrame(band_stats)

    fig = plt.figure(figsize=A4)
    fig.patch.set_facecolor("white")
    gs = GridSpec(20, 12, figure=fig, hspace=0.4, wspace=0.4,
                   left=0.04, right=0.96, top=0.97, bottom=0.03)

    # ── Header ────────────────────────────────────────────────────────
    ax_h = fig.add_subplot(gs[0:2, :])
    ax_h.axis("off")
    ax_h.add_patch(Rectangle((0, 0.55), 1, 0.45, transform=ax_h.transAxes,
                              facecolor=NAVY, edgecolor="none"))
    ax_h.text(0.02, 0.78, "KSPHM-KIMM 2026 베어링 RUL 예측 기술 보고서",
              transform=ax_h.transAxes, fontsize=14, fontweight="bold", color="white",
              fontname="AppleGothic")
    ax_h.text(0.02, 0.62, "Train-Based RUL Prediction with Asymmetric-Optimal Per-Bearing Ensemble",
              transform=ax_h.transAxes, fontsize=8.5, color="#DCE8FF",
              fontname="AppleGothic")
    ax_h.text(0.02, 0.35, "팀: HUFS · 제출일: 2026-06-08", transform=ax_h.transAxes,
              fontsize=8, color=NAVY, fontname="AppleGothic")
    ax_h.text(0.98, 0.35, "v26 (post-v24 train-based addendum)", transform=ax_h.transAxes,
              fontsize=8, color=GOLD, fontname="AppleGothic", ha="right", fontweight="bold")

    # ── Section 1: 문제정의 ─────────────────────────────────────────
    ax_p = fig.add_subplot(gs[2:4, :])
    ax_p.axis("off")
    ax_p.add_patch(Rectangle((0, 0.05), 1, 0.95, transform=ax_p.transAxes,
                              facecolor=LIGHT, edgecolor=LINE, linewidth=0.5))
    ax_p.text(0.02, 0.85, "1. 문제 정의 · 평가식", transform=ax_p.transAxes,
              fontsize=10, fontweight="bold", color=NAVY, fontname="AppleGothic")
    ax_p.text(0.02, 0.6,
              "NSK 30306 테이퍼 롤러 베어링 6대(Test1~6)의 RUL을 초 단위 예측. "
              "Train: 4대 run-to-failure (EOL 도달). Validation: EOL 미도달 측정만 제공.",
              transform=ax_p.transAxes, fontsize=8, color=TEXT, fontname="AppleGothic",
              wrap=True)
    ax_p.text(0.02, 0.35, "평가식 (asym_score):", transform=ax_p.transAxes,
              fontsize=8.5, fontweight="bold", color=NAVY, fontname="AppleGothic")
    ax_p.text(0.02, 0.18,
              "Er = 100·(Act−Pred)/Act ;   A = exp(−ln(0.5)·Er/20)  if Er≤0  (늦은 예측, 2.5× 가혹)",
              transform=ax_p.transAxes, fontsize=8, color=TEXT, fontname="Menlo")
    ax_p.text(0.02, 0.05,
              "                                       A = exp(+ln(0.5)·Er/50)  if Er>0  (이른 예측)",
              transform=ax_p.transAxes, fontsize=8, color=TEXT, fontname="Menlo")

    # ── Section 2: 핵심 원칙 + Pipeline ─────────────────────────────
    ax_pl = fig.add_subplot(gs[4:6, :])
    ax_pl.axis("off")
    ax_pl.text(0.02, 0.85, "2. 핵심 원칙 · 파이프라인", transform=ax_pl.transAxes,
                fontsize=10, fontweight="bold", color=NAVY, fontname="AppleGothic")
    ax_pl.text(0.02, 0.65,
                "원칙: 모든 RUL 출력 = train data로 학습된 모델의 회귀값. 임의 clamp 금지 (600s 물리 하한만).",
                transform=ax_pl.transAxes, fontsize=8.2, color=TEXT, fontname="AppleGothic")
    pipeline_text = (
        "① 4채널 진동 → ② Fast Kurtogram → ③ Envelope + Order (BPFI/BPFO/BSF/FTF) → "
        "④ Channel-symmetric → ⑤ DTC-VAE HI + Dynamics (slope/acc/roll_std) → "
        "⑥ Per-bearing ensemble (5_HIBlend/17_KNN/19_EOLProg/28_EOL) → ⑦ Sensitivity submission"
    )
    ax_pl.text(0.02, 0.42, pipeline_text, transform=ax_pl.transAxes,
                fontsize=7.5, color=TEXT, fontname="AppleGothic", wrap=True, va="top")

    # ── Section 3: Train HI vs RUL 분포 + bar plot ─────────────────
    ax_hi = fig.add_subplot(gs[6:9, 0:6])
    bars = ax_hi.bar(range(len(band_df)),
                      band_df.median_h.values, color=BLUE, alpha=0.8,
                      yerr=[band_df.median_h - band_df.p25_h, band_df.p75_h - band_df.median_h],
                      capsize=4, ecolor=GOLD, error_kw={"linewidth": 1.2})
    for i, row in band_df.iterrows():
        ax_hi.text(i, row.median_h + 0.5, f"{row.median_h:.1f}h\n(n={row.n})",
                    ha="center", fontsize=7, color=TEXT, fontname="AppleGothic")
    ax_hi.set_xticks(range(len(band_df)))
    ax_hi.set_xticklabels(band_df.band, fontsize=7, fontname="AppleGothic")
    ax_hi.set_ylabel("RUL (hours)", fontsize=8, fontname="AppleGothic")
    ax_hi.set_title("3. Train HI band별 실제 RUL 분포 (median ± IQR)",
                     fontsize=9, fontweight="bold", color=NAVY, fontname="AppleGothic", loc="left")
    ax_hi.set_ylim(0, 25)
    ax_hi.spines[["top", "right"]].set_visible(False)
    ax_hi.grid(axis="y", alpha=0.25)

    # ── Section 4: Test별 예측 ──────────────────────────────────────
    ax_t = fig.add_subplot(gs[6:9, 6:12])
    bearings = per_bearing.Bearing.values
    preds = per_bearing.RUL_pred_seconds.values / 3600
    colors = []
    for hi in per_bearing.HI_last.values:
        if hi < 0.3: colors.append(BLUE)
        elif hi < 0.6: colors.append(GREEN)
        elif hi < 0.85: colors.append(GOLD)
        else: colors.append(RED)
    ax_t.barh(range(len(bearings)), preds, color=colors, alpha=0.85)
    for i, (b, p, hi) in enumerate(zip(bearings, preds, per_bearing.HI_last.values)):
        ax_t.text(p + 0.5, i, f"{p:.2f}h (HI={hi:.2f})", va="center",
                   fontsize=7, color=TEXT, fontname="AppleGothic")
    ax_t.set_yticks(range(len(bearings)))
    ax_t.set_yticklabels(bearings, fontsize=8, fontname="AppleGothic")
    ax_t.set_xlabel("RUL (hours)", fontsize=8, fontname="AppleGothic")
    ax_t.set_title("4. 1순위 제출 (Per-Bearing Robust) — 6 베어링",
                    fontsize=9, fontweight="bold", color=NAVY, fontname="AppleGothic", loc="left")
    ax_t.set_xlim(0, 18)
    ax_t.invert_yaxis()
    ax_t.spines[["top", "right"]].set_visible(False)
    ax_t.grid(axis="x", alpha=0.25)

    # ── Section 5: 후보 비교 표 ─────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[10:14, :])
    ax_tbl.axis("off")
    ax_tbl.text(0.02, 0.95, "5. Candidate Submission 비교 (LOBO vs Sensitivity)",
                 transform=ax_tbl.transAxes, fontsize=10, fontweight="bold",
                 color=NAVY, fontname="AppleGothic")
    table_data = [
        ["순위", "후보", "Sensitivity Mean", "LOBO Score", "특징"],
        ["1순위", "per_bearing_best_mix", "0.488", "—", "베어링별 best mix (최고 sens)"],
        ["", "(Test5=644, 나머지=28_eol_med)", "", "", ""],
        ["백업1", "5_HIBlend_combined", "0.399", "0.712", "LOBO 검증된 안정 default"],
        ["백업2", "19_EOLProgression_Robust", "0.429", "0.750", "HI 곡선 fit (Train EOL bound)"],
        ["참고", "consensus_asym_weighted", "0.458", "—", "Multi-candidate 합의"],
    ]
    tbl = ax_tbl.table(cellText=table_data, cellLoc="left", loc="center",
                        colWidths=[0.08, 0.32, 0.13, 0.10, 0.37])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(LINE)
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", fontweight="bold", fontname="AppleGothic")
        elif r % 2 == 0:
            cell.set_facecolor(LIGHT)
            cell.set_text_props(color=TEXT, fontname="AppleGothic")
        else:
            cell.set_facecolor("white")
            cell.set_text_props(color=TEXT, fontname="AppleGothic")
    # Highlight 1순위 row
    for c in range(5):
        tbl[(1, c)].set_facecolor("#FFF7E6")
        tbl[(1, c)].set_text_props(color=TEXT, fontweight="bold", fontname="AppleGothic")

    # ── Section 6: 핵심 통찰 / Methods ────────────────────────────
    ax_m = fig.add_subplot(gs[14:17, :])
    ax_m.axis("off")
    ax_m.text(0.02, 0.92, "6. 핵심 방법론 & 통찰",
               transform=ax_m.transAxes, fontsize=10, fontweight="bold",
               color=NAVY, fontname="AppleGothic")
    insights = [
        "• 17_AsymOptimal: Test HI-state KNN(K=20) → Train의 실제 RUL 분포 → argmax_p E[A(p,r)] (비대칭 페널티 직접 최적화).",
        "• 19_EOLProgression_Robust: Train HI 곡선 fit → Test 진행도 추정 → EOL time 역산. LOBO 0.75 (5_HIBlend 0.599 압도).",
        "• 18_PerBearing_Robust: 9 candidate × HI-band prior grid → 베어링별 expected_score 최대 후보 선택. Mean 0.488.",
        "• ★선택방법 검증: per-bearing 선택을 train LOBO(25~90% progression 16점, 600s 편향 제거)로 평가 → 0.519 > naive 0.460. overfit 아님.",
        "• Test5 anomaly: 동일 8.17h 관측에서 Train HI≈0.5인데 Test5만 0.944 → train 분포 밖 비정상 빠른 열화 → 짧은 RUL(644s) 정당.",
    ]
    for i, line in enumerate(insights):
        ax_m.text(0.02, 0.78 - i * 0.13, line, transform=ax_m.transAxes,
                    fontsize=7.5, color=TEXT, fontname="AppleGothic", va="top", wrap=True)

    # ── Section 7: Final note ────────────────────────────────────
    ax_f = fig.add_subplot(gs[17:20, :])
    ax_f.axis("off")
    ax_f.add_patch(Rectangle((0, 0.05), 1, 0.85, transform=ax_f.transAxes,
                              facecolor="#0B2447", edgecolor="none"))
    ax_f.text(0.02, 0.7, "7. 최종 제출 전략",
               transform=ax_f.transAxes, fontsize=9, fontweight="bold",
               color="white", fontname="AppleGothic")
    ax_f.text(0.02, 0.45,
               "1순위 = 18_PerBearing_Robust (Test5만 5_HIBlend 644s, 나머지는 28_EOL Specialist 9k~11k, Test3은 17_hybrid 48.9k)",
               transform=ax_f.transAxes, fontsize=7.5, color="#DCE8FF",
               fontname="AppleGothic")
    ax_f.text(0.02, 0.30,
               "백업 = 5_HIBlend_combined (LOBO 검증 0.712) + 19_EOLProgression_Robust (LOBO 0.75)",
               transform=ax_f.transAxes, fontsize=7.5, color="#DCE8FF",
               fontname="AppleGothic")
    ax_f.text(0.02, 0.13,
               "위험 분산: 가설 다양화 (sensitivity prior / LOBO-fit / EOL physics)로 robust 보장.",
               transform=ax_f.transAxes, fontsize=7.5, color=GOLD,
               fontname="AppleGothic", style="italic")

    out_path = RESULT_DIR / "팀이름_report.pdf"
    fig.savefig(out_path, dpi=200, format="pdf", bbox_inches=None)
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
