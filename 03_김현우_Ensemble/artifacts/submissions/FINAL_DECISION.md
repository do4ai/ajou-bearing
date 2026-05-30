# 최종 제출 의사결정 — KSPHM-KIMM 2026 Bearing RUL

> 모든 후보는 train-based (임의 clamp 없음). 600s 물리 하한만 적용.
> 평가: 정확도30% + 우수성20% + 발표30% + 창의성10% + 합리성10%. 비대칭 페널티(늦은 2.5×).

## 4 후보 × 6 베어링 RUL (시간)

| Bearing | HI | 1순위 (per_bearing) | 1순위-cons (β0.95) | 백업1 (5_HIBlend) | 백업2 (19_EOLProg) |
|---------|-----|--------------------|--------------------|-------------------|--------------------|
| Test1 | 0.46 | 2.80h | 2.66h | 8.90h | 6.38h |
| Test2 | 0.50 | 3.06h | 2.90h | 9.32h | 6.67h |
| Test3 | 0.16 | 13.58h | 12.90h | 1.79h | 14.67h |
| Test4 | 0.45 | 2.65h | 2.52h | 3.92h | 6.08h |
| **Test5** | **0.94** | **0.18h** | **0.17h** | **0.18h** | **0.86h** |
| Test6 | 0.41 | 2.85h | 2.71h | 3.23h | 14.67h |

## 후보별 검증 점수

| 후보 | Sensitivity mean | Sensitivity worst | LOBO | 성격 |
|------|------------------|-------------------|------|------|
| 1순위 (per_bearing) | **0.488** | 0.401 | — | HI-band prior 최적, mid-life는 EOL specialist |
| 1순위-cons (β0.95) | 0.488 | **0.423** | — | worst-case robust (비대칭 보수) |
| 백업1 (5_HIBlend) | 0.399 | 0.299 | **0.712** | LOBO 검증된 안정 default |
| 백업2 (19_EOLProg) | 0.429 | 0.294 | **0.750** | EOL physics (HI 곡선 fit) |

## 핵심 의사결정 논리

### 왜 1순위 = per_bearing_robust?
- 6 베어링 sensitivity 평균 최고 (0.488). 각 베어링에서 train-based 후보 중 비대칭 기대값 최대.
- Test5는 5_HIBlend(644s) — train 분포 밖 outlier(8.17h에 HI=0.944)라 짧은 예측 정당.
- Test3은 17_hybrid(48900s) — HI=0.16으로 정상보다 느려 긴 RUL.
- mid-life(Test1,2,4,6)는 28_EOL specialist (9~11k).

### ★ 선택 방법 자체의 LOBO 검증 (24_ValidateSelectionMethod)
- **리스크**: 1순위는 sensitivity 0.488이지만 HI-band prior 가정에 overfit 우려 (LOBO 미검증).
- **검증**: train 베어링을 held-out으로 두고 동일 per-bearing 선택 로직을 25/50/75/90% progression 시점에 적용 (16 평가점, 600s 편향 없음).
- **결과**: selection 방법 LOBO asym **0.519** vs naive median 0.460 (+0.059) vs q35 0.466.
- → **per-bearing 선택이 단순 후보 대비 generalize 확인. overfit 아님.** (oracle 0.667 대비 gap -0.148, 개선 여지는 있으나 방법은 건전.)
- midlow band 0.622 (강), high band 0.478, midhigh 0.388 (약점 구간).
- **추가 stress-test (26_SelectionRuleSearch)**: 12개 대안 규칙(fixed quantile/blend/shrink λ)을 동일 16 LOBO점에서 비교 → 최고 in-sample(shrink_0.5, +0.051)은 Train3 단독 아티팩트(4 fold 중 2승, 16점 중 8승)로 기각. oracle gap은 4-베어링 한계의 비가역 불확실성 → **현 band_sens가 robust-optimal, 1순위 불변 확정.**

### 왜 백업을 다른 가설로?
- **위험 분산**: 1순위(sensitivity prior) / 백업1(LOBO train-fit) / 백업2(EOL physics).
- 세 관점이 독립적이라 한 가설이 틀려도 다른 후보가 방어.
- LOBO와 Sensitivity가 anti-correlated → 어느 한쪽만 믿으면 위험.

### Test5 — 승부처
- 모든 train-based 메서드가 짧은 RUL에 수렴 (644~3082s).
- 근거: HI=0.944(최고) + 8.17h에 train 어느 베어링보다 빠른 열화(anomaly) + DTW Train1 idx124(1200s) 최근접.
- 1순위/백업1 = 644s (공격), 백업2 = 3082s (약간 보수).

## 제출 권장 (예비 6/1~6/5)

1. **예비 1차**: `HUFS_validation_1순위.xlsx` → 점수 확인
2. **예비 2차** (점수 보고 조정): 1차가 낮으면 `HUFS_validation_백업2.xlsx` (긴 RUL 가설) 또는 `_conservative` (β0.95)

## 최종 제출 (6/8)

- 예비 결과 기반 best 1개 선택.
- 예비에서 1순위가 잘 나오면 → 1순위 유지.
- 예비에서 mid-life Test가 과소예측으로 판명되면 → 백업2(긴 RUL) 또는 백업1(5_HIBlend).

## 제출물 체크리스트

- [x] HUFS_validation_*.xlsx (4 후보, 모두 6행/RUL≥600/NaN없음 검증)
- [x] HUFS_code.zip (4.0MB 압축, 174 files/54 py, REPRODUCE.md, bit-exact 재현 검증)
- [x] HUFS_report.pdf (A4 1페이지, 값·출처 submission과 일치 — iter30 감사)
- [x] PPT (v24 38장 + v26 addendum 14장 = 52장)
- [x] PRESENTATION_SCRIPT.md (5분 + Q&A) + REHEARSAL_GUIDE.md
- [x] pre-flight 게이트 PASS (`python3 tools/preflight_check.py HUFS`)
- [ ] 팀명 최종 확정 후 모든 파일명 통일 (현 HUFS 가정)
