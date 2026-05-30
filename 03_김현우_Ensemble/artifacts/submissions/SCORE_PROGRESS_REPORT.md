# 챌린지 스코어 기준 실험 경과 보고 (03_김현우_Ensemble)

> 작성: 2026-05-29 (ralph iter44). 평가축 = 정확도30 + 우수성20 + 발표30 + 창의성10 + 합리성10.
> **전제**: Test1~6 정답 비공개 → 실제 챌린지 점수 미확인. 아래 수치는 전부 **train LOBO 대리지표**.

## ① 정확도 (30%) — asym_score 기준 ★핵심
> asym_score: 늦은 예측 2.5× 가혹. 1.0 완벽. 6베어링 평균.
> 경쟁 기준선: 01팀 0.454 / 02팀 0.422 → 후보군이 LOBO 기준 상회.

| 후보 | 방법 | LOBO 대리점수 | 측정 프로토콜 |
|---|---|---|---|
| 백업1 (5_HIBlend) | TFT+BiLSTM HI-blend | 0.712 | combined(full+last) |
| 백업2 (19_EOLProg) | HI곡선 fit 역산 | 0.75 | progression |
| 1순위 (18_PerBearing) | per-bearing 비대칭 선택 | 0.519 (>naive 0.46) | 16점 unbiased |
| conservative (β0.95) | 1순위 보수 틸트 | worst-case 0.40→0.42 | — |

⚠️ **위 표는 깔끔한 순위가 아님** — 각 숫자가 서로 다른 평가 프로토콜에서 나와 직접 비교 불가.

**신규(iter46) 독립 물리 방법** — `RUL=elapsed×(1−HI)/HI` (열화 평균율): train progression 16점 asym **0.586** = 단순 train 방법 중 최고(B 0.519/A 0.508/HI-fit 0.552 상회). 같은 16점 프로토콜이므로 B와 직접 비교 가능 → **+0.067**. raw 제출보다는 트랙 disambiguation 증거로 사용(EOL 가속 무시로 과대 경향). 상세 `docs/TRACK_RECONCILIATION.md` §9.

### 두 트랙 정면 충돌 (정확도 핵심 이슈)
제출 후보가 두 갈래(B=public-optimal, A=anti-overfit final)로 갈라져 있었고, 같은 잣대로 공정 비교:

| 관점 | 우위 | 근거 |
|---|---|---|
| near-EOL (수명 끝점) | **B (1순위)** | A anchor held-out EOL 점수 ~0, EOL gate 발화 1/4뿐 (iter42) |
| mid-life (HI-band prior) | **A (finaltest_robust)** | mid-life 4베어링 평균 A=0.508 vs B=0.378 (iter43) |

→ 두 관점이 정반대 = **LOBO↔Sensitivity anti-correlation 정량 확인**. 트랙 선택 = "HI가 베어링 간 transfer 하는가" 베팅. 현재는 **양쪽 모두 후보로 보유(헤지)**.

**결론**: 후보들은 경쟁팀(0.42~0.45)을 LOBO 상회하나, **실제 Test 점수를 좌우할 1순위 트랙 확정은 채점 대상(public vs 비라벨 최종셋)에 의존 → 미확정.** (상세: [TRACK_RECONCILIATION.md](../../docs/TRACK_RECONCILIATION.md))

## ② 우수성 (20%)
- Train-based 메서드 8종 + anti-overfit 계열(26/27 + DA).
- **임의 clamp 전무**: train 분포 내 선택 + 600s 물리 하한 + β 곱셈 보정만.
- code.zip **bit-exact 재현 검증 완료** (raw data 없이 1순위 예측 정확 복원).

## ③ 발표 (30%) — 최대 배점
- PPT 53장 (v24 38 + v26 addendum 15, conformal 포함) → **단일 PDF 병합 완료**: `KSPHM_KIMM_RUL_full_53slides_HUFS.pdf`.
- 5분 스크립트 + 리허설 가이드 + Q&A 8.

## ④ 창의성 (10%)
5개 차별점(과장 금지): ①평가식 비대칭성 점추정까지 일관 반영(argmax E[A]), ②per-bearing HI-band regime 선택, ③train-only+clamp 폐기, ④Test5 HI-velocity anomaly, ⑤LOBO 잔차 conformal 구간.

## ⑤ 합리성 (10%)
- LOBO 4-fold + sensitivity 이중 검증, overfit 규칙 적극 탐색 후 기각(iter29).
- Conformal median Er +23.4% (구조적 보수=2.5× late 정합).
- 두 트랙 정합 + 양방향 공정비교 = 정직한 불확실성 서술.

## 한 줄 요약
**우수성·발표·창의성·합리성은 사실상 완성. 정확도축 LOBO 대리점수는 경쟁팀 우위지만, 두 트랙 중 최종 1순위 확정이 미해결** — 채점 대상(public Test1~6 vs 별도 비라벨 최종셋) 확인 필요.

## 사용자 결정 대기
1. 채점 대상 확정 → 1순위 B↔A 결정 (TRACK_RECONCILIATION §4)
2. 팀명 확정 → `python3 tools/preflight_check.py <팀명>`
3. 6/1 예비 제출 / git commit (승인 시)
