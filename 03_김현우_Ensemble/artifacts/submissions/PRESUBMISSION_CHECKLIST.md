# KSPHM-KIMM 2026 예비 제출 체크리스트 (iter57 갱신)

> **확정 사실(공식 2026 챌린지 페이지 검증, iter57):** 제공 데이터는 **Train Set + Validation Set 둘뿐**.
> **채점 대상 = 검증용 데이터 셋 = Test1~6** (별도/비공개 최종 테스트셋 **없음**).
> 일정: **예비 6/1(월)~6/5(금)** → **최종 6/8(월)** → 결과발표 6/9 → 발표평가 6/25.
> ⚠ 공지(260527): **6월 1일 이전 제출 파일은 채점 제외·삭제**. 결과 제출 링크로 업로드.

---

## 0. ★ 이번 챌린지의 채점 구조 (가장 중요 — 전략의 토대)

| 단계 | 시점 | 무엇으로 채점 | 결과 |
|---|---|---|---|
| **예비 제출** | 6/1~6/5 | Validation(Test1~6) RUL 예측 → asym_score | **중간 순위 공개** (실측 점수 피드백!) |
| **최종 제출** | 6/8 | 같은 Validation 예측 (확정본) | **최종 순위 = 이 점수 기준** |
| 예선 심사 | ~6/8 | 정확도 + 창의성 + 합리성 (Report 포함) | **상위 7팀 본선 진출** |
| 본선 심사 | 6/25 | 예선 3기준 + 우수성 + 논리성 (발표) | 7팀 최종 등수 |

**핵심 전략적 함의:**
1. **예비 제출에 실측 점수(중간 순위)가 나온다** → 후보를 블라인드로 고를 필요 없음.
   **6/1~5에 가장 강한 후보를 올려 실제 asym_score를 받고, 그 신호로 6/8 최종을 확정.**
2. Validation은 **라벨 비공개** → 점수는 받지만 정답은 모름. 라벨 과적합은 불가, 단 점수 신호로 후보 선택은 가능.
3. **예선 게이트가 정확도뿐 아니라 창의성·합리성도 본다** → Report PDF·창의성 narrative도 6/8 전 완비 필수(본선 진출 조건).

---

## 1. 제출 후보 (artifacts/submissions/) — 가설별 정리

베어링 HI(진행도): Test1=0.46, Test2=0.50, Test3=0.16, Test4=0.45, **Test5=0.94(EOL 임박)**, Test6=0.41.

### ★ FLAGSHIP (메인, 재-base) — p* 비대칭점수-최적 점추정
| 파일 | T1 | T2 | T3 | T4 | T5 | T6 | 근거 |
|---|---|---|---|---|---|---|---|
| **`HUFS_validation_pstar.xlsx`** (37_PStar) | 29377 | 27574 | 65379 | 32394 | **1200** | 41400 | `p*=argmax_p E[asym(p,R)]` over train HI-KNN(K=20). 헤드라인을 *실제로* 구현하는 엔진. LOBO 0.5377(fold 정직공개). 전 값 train 이웃 support 내 |
| `HUFS_validation_pstar_conservative.xlsx` | 28496 | 26747 | 63418 | 31422 | 1164 | 40158 | p* × β0.97 (23_beta_sweep robust_mean 지배) |

> **왜 flagship 재-base:** 멀티에이전트 검증+코드 확인 결과, 구 1순위 ①(`per_bearing_robust.py`)은 9-벡터 **메타-셀렉터**라 발표 헤드라인("asym 직접최적화")과 절차 불일치(narrative seam). p\*가 그 헤드라인을 실제로 구현. 상세 = `BEST_METHOD_SELECTION.md`.
> **caveat:** mid-life LONG은 HI-transfer **베팅**(Train3 fold 0.335 붕괴), T6=41400은 severity 증거(짧음)와 충돌 → **6/1~5 예비로 1비트 실측 판정**. ①은 그 **대조군(Day2)**.

### A) "mid-life 짧음(~10k)" 가설 — 대조군/평가식 메타선택 계열

### A) "mid-life 짧음(~10k)" 가설 — 평가식 직접최적화 계열
| 파일 | T1 | T2 | T3 | T4 | T5 | T6 | 근거 |
|---|---|---|---|---|---|---|---|
| **`HUFS_validation_1순위.xlsx`** (18_PerBearing) | 10067 | 10998 | **48900** | 9545 | **644** | 10275 | 베어링별 HI-band asym-최적 선택. Sensitivity mean 0.488 |
| `HUFS_validation_1순위_conservative.xlsx` (β=0.95) | 9564 | 10448 | 46455 | 9068 | 612 | 9761 | 1순위 × 0.95 곱셈 보정 (2.5× late 페널티 worst-case 완화) |

### B) "mid-life 김(~32k)" 가설 — 안정 anchor 계열
| 파일 | T1 | T2 | T3 | T4 | T5 | T6 | 근거 |
|---|---|---|---|---|---|---|---|
| `HUFS_validation_백업1.xlsx` (5_HIBlend combined) | 32035 | 33556 | 6449 | 14113 | 644 | 11641 | **LOBO Combined 0.712** — 가장 검증된 안정 모델(anchor) |
| `HUFS_validation_finaltest_robust.xlsx` (26_FinalRobust) | 32035 | 33556 | 6449 | 14113 | 644 | **3000** | 5_HIBlend anchor + LOBO-frozen EOL gate(T6 발화). ※ iter57: 별도-최종셋 전제 반증 → **진단/대안용으로 강등** |
| `HUFS_validation_finaltest_T3fix.xlsx` | 32035 | 33556 | **48900** | 14113 | 644 | 3000 | 위 robust에서 **T3만 교정**(6449→48900; 저HI=긴RUL, 5-way 증거). A 채택 시 식별오류 보정판 |

### C) 물리 EOL-progression 계열
| 파일 | T1 | T2 | T3 | T4 | T5 | T6 | 근거 |
|---|---|---|---|---|---|---|---|
| `HUFS_validation_백업2.xlsx` (19_EOLProgression) | 22969 | 24000 | 52800 | 21906 | 3082 | 52800 | HI 곡선 fit→EOL 역산. LOBO mean **0.750**(단 Train1/2/4 perfect·Train3 fold 붕괴로 부풀려진 수치) |

> **두 가설의 정면 충돌점 = mid-life(T1/2/4):** A계열 ~32k vs B계열 ~10k. T3은 1순위/T3fix/백업2가 길게(48900~52800), robust/백업1만 짧게(6449). **iter46~49 물리·HI-prior 5-way 증거는 mid-life "긴 쪽"을 약간 지지**하나 n=4 부트스트랩상 비결정적 → **예비 제출 실측으로 가린다.**
> ⚠ `팀이름_*.xlsx` 3개는 옛 네이밍 잔존본(HUFS_와 값 동일) — 최종 전 정리 권장.

**왜 이게 후보인가 (공통):** 6종 전부 **train-based only**(임의 clamp 無, 600s 물리하한·β 곱셈보정만), 모두 preflight PASS(6행·RUL≥600·NaN無), 문서화 메서드와 **bit-exact 일치**. 서로 다른 가설(평가식최적화 / 안정anchor / 물리)로 **위험 분산**.

---

## 2. 예비 제출 실행 절차 (6/1~6/5)

- [ ] **팀명 확정** → `python3 tools/preflight_check.py <팀명>` 으로 6후보 재검증(전 PASS 확인)
- [ ] **6/1 1차 업로드**: 가장 강한 후보를 결과 제출 링크에 올림 (`<팀명>_validation.xlsx`)
      - 권장 1차: **1순위(B)** — 평가식 직접최적화, 가장 공격적·metric 정합
- [ ] **중간 순위(실측 점수) 확인** → 기록
- [ ] **2차 비교 업로드(가능하면)**: `finaltest_T3fix`(A계열 mid-life 김) 또는 `백업1` 올려 점수 비교
      → mid-life "짧음 vs 김" 가설을 **실측으로 판정**
- [ ] **6/5까지** 최고 점수 후보 확정 → 6/8 최종 제출본으로 락인
- [ ] ⚠ **6/1 이전 업로드 금지** (채점 제외·삭제 대상)

## 3. 최종 제출 (6/8) — 3종 모두 업로드

- [ ] `<팀명>_validation.xlsx` — 예비 점수 기준 최고 후보
- [ ] `<팀명>_code.zip` — `bash tools/build_code_zip.sh` (현재 64py/195files, bit-exact 재현 검증됨)
- [ ] `<팀명>_report.pdf` — A4 1p (1순위 narrative 확정 필요 — §아래)
- [ ] 최종 `python3 tools/preflight_check.py <팀명>` PASS

## 4. 제출물 무결성 (preflight 게이트)

- ✅ 6 validation 후보: 6행 / RUL≥600s / NaN無 / 베어링명 일치
- ✅ code.zip: REPRODUCE.md + shared/utils + feature CSV 포함, full-chain bit-exact 재현
- ✅ report.pdf: 유효 헤더, 값·출처 submission 일치 (레이아웃 QA 완료)
- ✅ 5종 source bit-exact 대조 (1순위↔18 / conservative↔23 / 백업1↔5_HIBlend / 백업2↔19 / finaltest↔26)

---

## 5. 발표·합리성·창의성 핵심 메시지 (예선 게이트 + 본선 50%)

1. **Train-Based Only** — 임의 숫자 박기 폐기, 600s 물리하한·β 곱셈보정만. 챌린지 정신 정합.
2. **비대칭 점수 직접 최적화** — `argmax_p E_R[A(p,R)]` over train HI-state 분포. 평가 metric을 objective로.
3. **Per-Bearing HI-band Selection** — LOBO 0.519 > naive 0.460 (generalize 입증, overfit 아님).
4. **Test5 anomaly** — 동일 8.17h에 train HI~0.48 vs Test5 0.944 → 비정상 급속열화 → 짧은 RUL(644) 정당.
5. **물리 열화율 + 2축 건강평가** — `RUL=elapsed×(1−HI)/HI` + HI(진행도)×energy(심각도)로 Test5·Test6 이상치 통합 설명. n=4 부트스트랩으로 우열 비결정적임을 정직 서술(과대정밀 회피).
6. **Conformal 구간** — LOBO 잔차 +23.4% 보수 bias가 2.5× late 페널티에 정합(점추정 불변·additive).

---

## 6. 남은 결정/액션

- [x] **flagship 확정 = p\*** (`HUFS_validation_pstar.xlsx`). ①은 예비 Day2 대조군으로 강등, 26(A)는 별도셋 전제 반증으로 대안 기록용.
- [ ] **팀명 확정** (현재 HUFS 가정) → `python3 tools/preflight_check.py <팀명>`
- [ ] **report.pdf를 p\* flagship narrative로 갱신** (현 report는 구 ① 기준 — 6/8 전 교체 필요)
- [ ] **6/1 예비**: Day1 p\* → Day2 ① 대조 → mid-life 1비트 판정 → 6/8 lock
- [ ] **git commit** (사용자 승인 시 / 본 작업분은 승인됨)
