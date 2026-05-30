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

    # Flagship = 앙상블(물리 avg-rate × 비대칭점수-최적 p*) 제출본
    flagship = pd.read_excel(RESULT_DIR / "42_blend_submission.xlsx")

    # Train HI vs RUL stats
    df = pd.read_csv(FEATURE_CSV).fillna(0)
    train = df[df.bearing.isin(TRAIN_NAMES)]
    hi_last = {b: float(df[df.bearing == b].HI.iloc[-1]) for b in flagship.Bearing}
    flagship["HI_last"] = flagship.Bearing.map(hi_last)

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
    ax_h.text(0.02, 0.62, "Train-Based Ensemble: Asymmetric-Score-Optimal Point Estimation × Physics Degradation-Rate",
              transform=ax_h.transAxes, fontsize=8.0, color="#DCE8FF",
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
        "④ Channel-symmetric → ⑤ DTC-VAE HI(건강지표) → "
        "⑥-a p* = argmax_p E[asym(p,R)] (HI-KNN K=20, 평가식 직접최적화)  +  "
        "⑥-b 물리 열화율 RUL = elapsed·(1−HI)/HI → ⑦ 두 추정기 고정 기하평균 앙상블"
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
    bearings = flagship.Bearing.values
    preds = flagship.RUL_pred_seconds.values / 3600
    colors = []
    for hi in flagship.HI_last.values:
        if hi < 0.3: colors.append(BLUE)
        elif hi < 0.6: colors.append(GREEN)
        elif hi < 0.85: colors.append(GOLD)
        else: colors.append(RED)
    ax_t.barh(range(len(bearings)), preds, color=colors, alpha=0.85)
    for i, (b, p, hi) in enumerate(zip(bearings, preds, flagship.HI_last.values)):
        ax_t.text(p + 0.4, i, f"{p:.2f}h (HI={hi:.2f})", va="center",
                   fontsize=7, color=TEXT, fontname="AppleGothic")
    ax_t.set_yticks(range(len(bearings)))
    ax_t.set_yticklabels(bearings, fontsize=8, fontname="AppleGothic")
    ax_t.set_xlabel("RUL (hours)", fontsize=8, fontname="AppleGothic")
    ax_t.set_title("4. Flagship 제출 (앙상블 avg-rate×p*) — 6 베어링",
                    fontsize=9, fontweight="bold", color=NAVY, fontname="AppleGothic", loc="left")
    ax_t.set_xlim(0, 19)
    ax_t.invert_yaxis()
    ax_t.spines[["top", "right"]].set_visible(False)
    ax_t.grid(axis="x", alpha=0.25)

    # ── Section 5: 후보 비교 표 ─────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[10:14, :])
    ax_tbl.axis("off")
    ax_tbl.text(0.02, 0.95, "5. 후보 비교 — 동일 held-out 점 공정 LOBO (+ 부트스트랩)",
                 transform=ax_tbl.transAxes, fontsize=10, fontweight="bold",
                 color=NAVY, fontname="AppleGothic")
    table_data = [
        ["역할", "방법", "공정 LOBO", "95% CI", "특징"],
        ["Flagship", "앙상블 = avg-rate × p* (geo)", "0.633", "[.57,.72]", "두 독립 추정기 고정 기하평균 (0-param)"],
        ["정확도 anchor", "물리 avg-rate", "0.600", "[.55,.65]", "RUL=elapsed·(1−HI)/HI — 정확도 최선"],
        ["metric anchor", "p* (asym-argmax)", "0.538", "[.40,.66]", "argmax_p E[asym] — seam-free·Test5 샤프"],
        ["백업(NN)", "5_HIBlend", "0.712*", "*별 프로토콜", "안정 TFT+BiLSTM anchor"],
    ]
    tbl = ax_tbl.table(cellText=table_data, cellLoc="left", loc="center",
                        colWidths=[0.13, 0.30, 0.11, 0.12, 0.34])
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
        "• p* (의사결정이론 점추정): Test HI의 Train HI-KNN(K=20) 실제 RUL 분포에서 argmax_p E[asym(p,R)] — 평가식을 목적함수로 한 점추정, 제출값=명시 목적함수 출력(seam 없음).",
        "• 물리 열화율 (avg-rate): RUL = 경과·(1−HI)/HI. 동일 held-out 점 공정 LOBO 0.600 = train 점추정기 중 정확도 최선.",
        "• ★앙상블: 두 독립 추정기(의사결정 × 물리) 고정 기하평균(0-param) → 공정 LOBO 0.633, 양쪽 robust 능가(P=0.74~0.88, 분산 감소).",
        "• 강건성·정직성: K∈[8,40] LOBO 평탄(0.019)·방향 K-불변. n=4라 CI 넓음 → '결정적' 아닌 'robust 우위'로 서술·예비로 확인.",
        "• Test5 anomaly: 8.17h에 Train HI≈0.48 vs Test5 0.944 → 급속 열화 → 짧은 RUL(앙상블 1254s, 같은 규칙서 자동).",
    ]
    for i, line in enumerate(insights):
        ax_m.text(0.02, 0.80 - i * 0.165, line, transform=ax_m.transAxes,
                    fontsize=7.3, color=TEXT, fontname="AppleGothic", va="top", wrap=True)

    # ── Section 7: Final note ────────────────────────────────────
    ax_f = fig.add_subplot(gs[17:20, :])
    ax_f.axis("off")
    ax_f.add_patch(Rectangle((0, 0.05), 1, 0.85, transform=ax_f.transAxes,
                              facecolor="#0B2447", edgecolor="none"))
    ax_f.text(0.02, 0.7, "7. 최종 제출 전략",
               transform=ax_f.transAxes, fontsize=9, fontweight="bold",
               color="white", fontname="AppleGothic")
    ax_f.text(0.02, 0.45,
               "Flagship = 앙상블(avg-rate × p*) : 23910 / 28454 / 48935 / 34245 / 1254 / 44812 초 (Test1~6).  전 값 train 이웃 support 내·600s 하한·임의 clamp 無.",
               transform=ax_f.transAxes, fontsize=7.2, color="#DCE8FF",
               fontname="AppleGothic")
    ax_f.text(0.02, 0.30,
               "Anchor = avg-rate(정확도 LOBO 0.600) · p*(metric/seam) ;  백업 = 5_HIBlend(LOBO 0.712).  예측 코드는 code.zip 단독 bit-exact 재현.",
               transform=ax_f.transAxes, fontsize=7.2, color="#DCE8FF",
               fontname="AppleGothic")
    ax_f.text(0.02, 0.13,
               "예비 제출(6/1~5) 실측으로 mid-life long/short·Test6 1비트 확정 후 6/8 최종 lock. 위험 분산 = 의사결정 × 물리 × NN 가설 다양화.",
               transform=ax_f.transAxes, fontsize=7.2, color=GOLD,
               fontname="AppleGothic", style="italic")

    out_path = RESULT_DIR / "팀이름_report.pdf"
    fig.savefig(out_path, dpi=200, format="pdf", bbox_inches=None)
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
