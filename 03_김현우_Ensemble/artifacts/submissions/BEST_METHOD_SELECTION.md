# 최고의 방법 선정 — 비대칭점수-최적 점추정 (p*)

> 멀티에이전트 종합심사(6축 매핑 → 3관점 패널 → 적대적 검증 → 종합) + 코드 직접 재확인 결과.
> 챌린지 3대 목표(파이널 스코어 최고점 / 과적합 방지 / 심사기준 정합)에 매핑한 단일 최고 방법.

## 결론

**방법:** 각 베어링의 마지막 HI에서 **train 전체의 HI-KNN(K=20) 이웃의 실제 rul_s 분포**를 잡고,
그 분포에 대해 **`p* = argmax_p E[asym_score(p, R)]`** 를 푼다.
평가식(비대칭, 늦은예측 2.5×)을 **그대로 목적함수**로 한 의사결정이론 점추정. 한 규칙에서 6값이 전부 도출.
train-only · 임의 clamp 無 · 600s 물리 하한 · β 곱셈보정만.

**구현/산출:** `experiments/17_AsymOptimal_TrainBased/p_star_estimator.py`
→ `artifacts/results/.../37_pstar_submission.xlsx` = `artifacts/submissions/HUFS_validation_pstar.xlsx` (flagship).

**flagship 제출값 (전부 train 이웃 support 내, ≥600s):**

| | Test1 | Test2 | Test3 | Test4 | Test5 | Test6 |
|---|---|---|---|---|---|---|
| HI | 0.46 | 0.50 | 0.16 | 0.45 | 0.94 | 0.41 |
| 이웃 RUL med | 36900 | 30900 | 72900 | 37500 | 3600 | 47400 |
| **p\* (초)** | **29377** | **27574** | **65379** | **32394** | **1200** | **41400** |
| E*(이웃 기대점수) | 0.732 | 0.678 | 0.727 | 0.693 | 0.503 | 0.751 |
| β0.97 보수 | 28496 | 26747 | 63418 | 31422 | 1164 | 40158 |

> **★ 최종 권장(iter62):** 아래 p\*는 **발표 seam-free·metric-alignment** 축 최선, 물리 **avg-rate**는 **정확도 LOBO 최선(0.600 > p\* 0.538, P=0.88)**. 둘은 경쟁이 아니라 **결합**: 고정 기하평균 앙상블 **avg×p\* = 0.633**이 둘 다 robust 능가(0-param·과적합無). → **단일 최고 방법 = 두 추정기 기하평균 앙상블** (`HUFS_validation_blend.xlsx` = 23909/28454/48934/34245/1254/44812). 상세 "종합(iter62)" 섹션. (n=4라 결정적 아님 → 6/1~5 예비 실측 확인.)

## ★ 선정의 핵심 — 코드로 확인한 결정적 사실

현재 커밋된 1순위(①, `per_bearing_robust.py`)의 **발표 헤드라인("asym 직접 최적화 argmax_p E[A]")과 실제 절차가 불일치**한다.
- ① 실제 코드(line 40): `best_c = max(means, key=means.get)` — **9개 사전계산 모델출력**(28_eol_med, 5_HIBlend 등) 중 sensitivity-grid 평균 최고를 고르는 **메타-셀렉터**. argmax_p E[asym]를 풀지 않는다.
- 그 헤드라인을 *실제로* 구현하는 엔진은 `midlife_headtohead.py`의 p\* (line 64–66 `grid[argmax E_neigh[asym]]`).
- → "stated objective가 제출 숫자를 만드는가?" Q&A에 ①은 "아니오", p\*는 "예". **발표 30% 축 최대 약점(narrative seam)을 p\* 채택으로 제거.**

## 3대 목표 매핑

**1) 파이널 스코어 최고점 (정확도 30%)**
- p\*는 베어링별 평가식 기대값을 직접 최대화(metric-as-objective). 같은 규칙에서 **Test5는 자동으로 짧게(1200, 근-EOL 이웃 분포의 비대칭-최적점)**, mid-life/저진행은 길게 도출.
- ⚠ mid-life LONG(~30k)·T3 LONG(65379)은 **"HI가 베어링 간 전이된다"는 가정 하의 최적**. 부트스트랩상 트랙 우열은 비결정(P승=0.42, CI 전면중첩). → **6/1~5 예비 리더보드로 mid-life long/short 1비트 실측 판정**(아래).

**2) 과적합 방지**
- **단일 생성규칙** → per-cell 수동조립 0 (⑤의 조립 표면 / ④의 단일 게이트 / ①의 9-벡터 메타선택 구조적 제거).
- **비순환**: p\*는 다른 제출후보(A/B)로 학습되지 않은 독립 train KNN 분포의 argmax → in-sample 과적합 없음.
- 모든 p\* ∈ [이웃 lo, 이웃 hi] + ≥600s (외삽·임의값 없음, 자동 검증 통과).
- 노출면 = HI-transfer 가정 **하나**뿐, 그것만 예비로 검증·공개. 부풀린 LOBO(⑥)·저신뢰 게이트(④) 미사용.

**3) 심사기준 정합**
- **발표 30%**: seam 제거(제출=목적함수 출력). 4대 메시지(train-based / asym 직접최적화 / per-bearing HI-band / Test5 anomaly)가 한 규칙에서 분절 없이 파생.
- **창의성 10%**: 의사결정이론 점추정 — 블랙박스 DL과 명확 차별, 라벨·코드 일치.
- **합리성 10%**: 아래 LOBO fold 분산·부트스트랩 CI 중첩을 헤드라인 정직성으로 제시.
- **우수성 20%**: 어려운 세 구간(EOL T5 / 저진행 T3 / mid-life)을 한 규칙이 자동 처리.

## LOBO 재현 (정직한 일반화 검증) — `37_pstar_lobo.csv`

held-out 베어링 progression에 p\* 규칙 적용(이웃은 나머지 3 train만 → out-of-bearing):

| held | fold mean asym |
|---|---|
| Train1 | 0.699 |
| Train2 | 0.543 |
| Train3 | **0.335** (HI-transfer 붕괴 — 정직 공개) |
| Train4 | 0.574 |
| **전체** | **0.5377** (range 0.364) |

→ LOBO 0.5377은 ⑥의 부풀린 0.750(Train3 fold 8e-15 아티팩트)보다 **정직**하고, HI-transfer 진단(0.552)과 정합. **mid-life long은 "검증된 사실"이 아니라 "예비로 판정할 베팅"**임을 이 분산이 보여준다.

## 정직한 caveat (반드시 이 프레임)
1. mid-life LONG = 베팅(HI-transfer가 Train3에서 붕괴). "검증된 사실" 주장 금지.
2. **T6=41400이 최대 약점 셀**: 2축 energy-severity(energy 23.3=train EOL 2배=숨은 급성)는 T6를 짧게(~3000) 봄 → p\*(이웃 argmax)와 충돌. 예비 Day3 실측 검증 필요.
3. E*가 E_A·E_B를 다 이기는 건 **같은 목적함수의 argmax라 당연(천장)** — 우월성 증거 아님. "최고 기대점수"가 아니라 "**가장 방어가능·과적합 없는 단일 원리**"로 주장.
4. β는 0.95가 아니라 **0.97**(23_beta_sweep robust_mean 0.4898 지배).

## 2축 조건화 변형 탐색 (iter59) — 정직한 음성 + 부수 수확

T6 약점을 손보정 없이 해소하려 **동일 p\* 엔진을 2축(HI × log-energy) KNN으로 조건화**(`p_star_2axis.py`, LOBO 재현 `39_pstar_2axis_lobo.csv`). 사전등록 채택기준 = "LOBO 평균 ≥ HI-only AND T6 원리적 해소".

| | Train1 | Train2 | Train3 | Train4 | 전체 | range |
|---|---|---|---|---|---|---|
| HI-only | 0.699 | 0.543 | **0.335** | 0.574 | 0.5377 | 0.364 |
| 2-axis | 0.523 | 0.545 | **0.544** | 0.540 | 0.5378 | **0.022** |

**판정 = flagship 미교체(HI-only p\* 유지).** 이유: (1) 평균 LOBO 동일(Δ+0.0001 → 정확도상 교체 근거 없음), (2) 2축이 **Test5를 1200→3036으로 늘림** — 가장 확신 높은 anomaly 셀을 악화.

**그러나 부수 수확 2가지(합리성·발표 자산):**
- **Train3 붕괴(0.335)가 2축에서 0.544로 복원** → "HI-transfer가 Train3에서 무너진다"는 약점이 **방법 실패가 아니라 under-conditioning 아티팩트**임을 입증. fold 분산 0.364→0.022로 급감. **Q&A 방어**: "HI 단독은 Train3에서 붕괴하나, 심각도(energy) 축을 더하면 0.54로 복원돼 붕괴가 조건화 부족 탓임을 보였다."
- **T6가 energy 축으로 41400→28254로 자연 하강** → 2축 severity 신호(숨은 급성)를 **argmax 프레임 안에서 정량 확인**(손보정 아님). T6 불확실성 caveat의 데이터 근거 강화.

## K 민감도 감사 (iter60) — flagship은 유일 knob(K)에 robust

p\*의 유일 하이퍼파라미터 K(HI-KNN 이웃 수)를 K∈{8,12,16,20,28,40}로 sweep(`p_star_ksweep.py`, `40_pstar_ksweep.csv`):

| 지표 | 결과 |
|---|---|
| LOBO 전체 | 0.5319~0.5505 (**range 0.019 = 평탄**) |
| 방향(SHORT/LONG) 불변성 | **6 베어링 전부 K-불변** (T1/2/3/4/6 항상 LONG, **T5 항상 SHORT**) |
| K-sweep 값대역(민감도 구간) | T1 28.8k~36.0k · T2 27.0k~30.6k · **T3 50.4k~73.2k** · T4 25.2k~36.6k · **T5 1200~2982** · T6 40.2k~46.2k |

**판정 = K-ROBUST.** 점추정의 **방향(=의사결정)은 K에 의존하지 않음**; K=20은 안정 평탄대.
- **Q&A 방어**: "왜 K=20? K 바꾸면?" → "K∈[8,40]서 LOBO 평탄(Δ0.019)·6셀 방향 전부 불변. K=20은 평탄대 중앙."
- **정직 disclose**: 절대 크기는 일부 셀(특히 **T3 50k~73k**) 변동 → K-sweep 대역을 그대로 **민감도 구간**으로 보고(과대정밀 회피). T5는 1200~2982로 전부 짧음(anomaly 견고).

## ★ 정확도 공정 재검증 (iter61) — 정직한 보정: 물리 avg-rate가 LOBO 정확도 1위

그동안 후보 LOBO 수치가 **서로 다른 프로토콜**에서 나와 비교 불가였음. 임의 progression 점에
적용 가능한 train-based 점추정기들을 **동일 held-out 점**에서 공정 채점(`unified_lobo_comparison.py`,
`41_unified_lobo.csv`, 4-베어링 부트스트랩 256 CI):

| 점추정기 | mean asym | 95% CI | Train3 |
|---|---|---|---|
| **avg-rate 물리** | **0.600** | [0.547, 0.646] | 0.518 |
| HI-regression | 0.598 | [0.374, 0.786] | 0.246 |
| KNN-q35 | 0.549 | [0.322, 0.717] | 0.217 |
| p\*-2axis | 0.538 | [0.528, 0.544] | 0.544 |
| **p\* (flagship)** | **0.538** | [0.395, 0.660] | 0.335 |
| KNN-median | 0.530 | — | 0.151 |

**P(p\* > avg-rate) = 0.12** → 물리 avg-rate(`RUL=elapsed×(1−HI)/HI`)가 p\*를 **공정 LOBO서 robust하게 능가**(88% 표본).

**정직한 보정:** 앞서 "p\* = 단일 최고"는 **발표 seam-free·창의성** 축에선 타당하나 **정확도(30%, 최대 단일축)에선 과장**이었음. 정확도-best train 점추정기는 **물리 열화율 모델**.
- **두 방법은 큰 베팅에서 일치**: mid-life LONG + Test5 SHORT. (avg-rate test = 19459/29361/36626/36201/**1310**/48506 ; p\* = 29377/27574/65379/32394/**1200**/41400)
- 차이는 **크기·EOL 처리**뿐: T3(p\* 65k vs avg 37k, 둘 다 long), **T6(p\* 41k vs avg 49k — 둘 다 LONG, 단 2축 severity는 짧음 주장 → T6 최대 미결)**. avg-rate는 EOL 가속 무시로 Test5를 1310(p\*는 1200으로 더 샤프).

**revised flagship 프레임 = 2 anchor, 예비가 판정:**
- **정확도 anchor: 물리 avg-rate** (`HUFS_validation_avgrate.xlsx`, LOBO 0.600·CI 최상위, 물리 해석=창의성 문헌 정합).
- **metric/seam anchor: p\*** (`HUFS_validation_pstar.xlsx`, asym 직접최적화·Test5 최샤프).
- 둘 다 seam-free·train-only·mid-life LONG → **6/1~5 예비에 둘 다 올려 실측 우열로 6/8 확정**. 사용자 #1=최종 스코어 기준이면 정확도상 avg-rate가 근소 우위, 발표 metric-alignment는 p\* 우위.

## ★★ 종합 (iter62) — 고정 기하평균 앙상블이 두 anchor를 모두 능가 (2-anchor 긴장 해소)

avg-rate vs p\*를 **고를** 필요 없음 — **고정(0-param) 기하평균**이 둘 다 robust하게 능가(`blend_estimators.py`, `42_blend_lobo.csv`):

| 방법 | mean asym | 95% CI | P(>avg-rate) |
|---|---|---|---|
| **avg×HIreg (geo)** | **0.636** | [0.560,0.712] | **0.88** |
| **avg×p\* (geo)** | **0.633** | [0.570,0.716] | 0.74 |
| avg×p\*×HIreg (geo) | 0.626 | [0.535,0.725] | 0.68 |
| avg-rate (최선 단일) | 0.600 | [0.547,0.646] | — |
| p\* | 0.538 | [0.395,0.660] | 0.12 |

**avg-rate를 포함한 모든 기하-블렌드가 단일 최선을 능가** → 체리픽 아닌 **분산 감소(variance reduction)** 의 robust 증거. 학습 weight 無(0-param) → 과적합 표면 없음. 블렌드 하한 CI(0.560)도 avg-rate 하한(0.547)보다 높음.

**→ 최종 단일 최고 방법 = 두 독립 train-based 추정기의 고정 기하평균 앙상블.** 권장 = **avg-rate × p\*** (`HUFS_validation_blend.xlsx`): 두 flagship anchor(물리 열화율 + 비대칭점수-최적 의사결정)를 결합해 가장 깔끔한 서사 + 정확도 0.633(P=0.74). (최고 LOBO는 avg×HIreg 0.636/P0.88이나 HI-reg는 단독 고변동·서사 약함 → 노이즈 내 동급이면 anchor 결합형 채택.)

**앙상블 제출값** (= 두 anchor 제출값의 베어링별 기하평균): **23909 / 28454 / 48934 / 34245 / 1254 / 44812**.
- Test3=48934(원 B와 사실상 일치), Test5=1254(짧음), mid-life·T6 LONG — 두 anchor의 방향 일치를 그대로 계승, 크기는 중앙값화.
- **정직 caveat**: n=4라 블렌드 CI도 넓고(0.570~0.716) avg-rate와 겹침 — "robust 우위(P=0.74~0.88)"이지 "결정적"은 아님. mid-life LONG·T6는 여전히 예비 실측 판정 대상.

## 예측구간 (iter64) — flagship 앙상블의 LOBO-잔차 conformal (점추정 불변·additive)

`conformal_ensemble.py`(`44_conformal_ensemble.csv`): 앙상블 규칙의 LOBO held-out Er(%) 경험분포 → 테스트 점추정에 split-conformal 유사 90% 구간 부여.
- **median Er = +9.0%** (구조적 보수=이른 예측 경향) → 2.5× late 페널티에 정합(asym-최적의 자연 귀결).
- 90% 구간(시간): T1 6.64[5.0,15.7] · T2 7.90[6.0,18.7] · T3 13.59[10.3,32.2] · T4 9.51[7.2,22.5] · **T5 0.35[0.26,0.83]** · T6 12.45[9.4,29.5].
- **Test5는 90% 상한도 0.83h(<1h)** → 짧은 RUL anomaly가 불확실성 하에서도 robust. 구간 폭 큼 = n=4 한계의 정직한 반영(과대정밀 회피). **점추정은 제출본 불변(순수 additive)**.

## 예비 리더보드(6/1~5) 결정 실험
- **Day1**: ★ **앙상블(avg×p\*)** 메인 + 정확도 anchor **avg-rate** + metric anchor **p\*** 동시 제출 → 앙상블이 실제로 단일을 이기는지 실측 확인.
- **Day1(구)**: 정확도 anchor **avg-rate**(LOBO 0.600) + metric anchor **p\*** 동시 제출 → 실측 우열 확보.
- **Day1(구)**: p\*(mid-life LONG) 제출 = 메인 백본 실측 기준선.
- **Day2**: 대조군 ①(mid-life SHORT ~10k) 제출. 두 벡터는 T3·T5 방향이 같아 **점수차 거의 전부가 mid-life long/short 식별**.
- p\* ≫ ① → LONG 확증 → 6/8 메인=p\* 동결. ① ≥ p\* → mid-life만 SHORT 재추정(엔진 동일, 가정만 데이터로 갱신).
- **가드레일**: per-cell 튜닝 금지(1비트만), 비결정적이면 일반화 견고한 p\* 유지, train-based 원칙 보호 위해 공개 disclose.

## 후보 위상 (재-base 후)
- **flagship(메인)**: `HUFS_validation_pstar.xlsx` (p\*) + `_pstar_conservative.xlsx` (×β0.97)
- **대조군(예비 Day2)**: `HUFS_validation_1순위.xlsx` (① 메타-셀렉터, mid-life short)
- **백업/대안**: 백업1(5_HIBlend) · 백업2(19_EOLProg) · finaltest_robust/T3fix(26, 별도셋 전제 반증으로 강등)
- 8 validation 후보 전부 preflight PASS, 7종 source bit-exact.
