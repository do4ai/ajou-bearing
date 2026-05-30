# 트랙 정합 (Track Reconciliation) — 두 제출 트랙의 현황 통합

> 작성: 2026-05-29 (ralph-loop iter 41). 사용자 지시 "랄프루프로 계속해서 고도화하고 있었어 현황 확인해" 에 대한 정합 보고.
> 목적: 같은 ralph-loop 연속 고도화 중 **두 갈래로 갈라진 제출 트랙**을 하나의 정확한 현황으로 합친다.

## 1. 두 트랙은 무엇이며 언제 만들어졌나 (타임스탬프 검증)

| 트랙 | 핵심 산출물 | mtime | 설계 의도 |
|------|------------|-------|----------|
| **A. Anti-Overfit Final 트랙** | `docs/FINAL_ANTI_OVERFIT_PROTOCOL.md`, `26_FinalRobust_LOBOFrozenSelector`, `27_EOLClassifier_LOBOCalibrated`, `17_ConditionalMADA`~`25_StageAware` (DA 계열) | **05-28 20:20~20:32** | **별도 공개될 비라벨 최종 테스트셋** 대비. LOBO-frozen 앵커(5_HIBlend) + EOL gate, public 베어링별 수동 보정 **금지** |
| **B. Public-Optimal 트랙** | `17_AsymOptimal_TrainBased/18_per_bearing_robust`, `19_EOLprog`, `23_β0.95`, `conformal_intervals`, `artifacts/submissions/HUFS_*` 패키지 | **05-29 02:18~14:10** | public Test1~6에 대해 **베어링별 비대칭 기대점수 최대(argmax_p E[A])** 선택. 제출 패키지(validation/code/report/PPT) 완성 |

→ A가 먼저(5/28 저녁), B가 나중(5/29). **B가 더 최신이지만, B를 만들 때 A의 핵심 통찰(최종 테스트셋 분리 → 베어링별 수동튜닝 위험)을 반영하지 않았다.** 두 트랙은 서로를 모른 채 각자 완결되었다 (어느 카탈로그 문서도 상대 트랙을 언급하지 않음 — 확인됨).

## 2. 두 트랙은 예측이 크게 다르다 (정합의 핵심)

| Bearing | HI-band | **B-1순위 (per-bearing asym-opt)** | **A-finaltest (5_HIBlend+EOL gate)** | 비고 |
|---------|---------|-----------------------------------|--------------------------------------|------|
| Test1 | 0.46 mid | 10,067s | 32,035s | **3.2× 차이** |
| Test2 | 0.50 mid | 10,998s | 33,556s | **3.1× 차이** |
| Test3 | 0.16 low | **48,900s** | **6,449s** | **7.6× 역전** (저 HI=초기=긴 RUL이면 B가 정합적) |
| Test4 | 0.45 mid | 9,545s | 14,113s | 1.5× |
| Test5 | 0.94 high | 644s | 644s | **유일하게 일치** (둘 다 EOL 판정) |
| Test6 | 0.41 mid | 10,275s | 3,000s | **3.4× 차이** |

전 후보 값표는 `artifacts/submissions/PREFLIGHT_MANIFEST.txt` 및 preflight `[4]` 일관성 검증 참조.

**해석**: 두 트랙이 합의하는 건 Test5(EOL)뿐. Test1/2/3/4/6은 정성적 방향까지 다르다. 이는 단순 calibration 차이가 아니라 **방법론 철학의 차이**:
- B는 public 베어링 각각에서 비대칭 점수를 최대화 → public 점수에 강하지만, 평가가 비라벨 최종셋이면 과적합 위험(A 프로토콜이 경고한 바로 그 패턴).
- A는 LOBO로 동결된 단일 규칙 → public 점수는 다소 낮을 수 있으나 미지 분포에 robust.

## 3. 정합 조치 (이번 iter에서 한 것)

1. **A의 최종-테스트 후보를 제출 패키지에 편입**: `artifacts/submissions/HUFS_validation_finaltest_robust.xlsx` 생성 (= 26_FinalRobust, 6행·RUL≥600·NaN無 검증 완료). 이전엔 제출 폴더에 B 계열 4종(1순위/conservative/백업1/백업2)만 있어 **최종-테스트 안전 후보가 부재**했다 → 해소.
2. **preflight 게이트 확장**: `tools/preflight_check.py [4]`에 `finaltest_robust ↔ 26_FinalRobust` bit-exact 일관성 검증 추가.
3. **이 문서**로 두 트랙을 하나의 현황으로 카탈로그화 (이전엔 어느 문서도 상호 참조 없음).

## 4. 미해결 — 사용자 결정 필요

**평가가 무엇에 대해 채점되는가**에 따라 1순위가 갈린다:
- **public Test1~6가 곧 최종 점수** → B(per-bearing asym-opt)가 1순위로 타당.
- **별도 비라벨 최종셋이 본 점수**(운영위 공지, A 프로토콜 전제) → A(finaltest_robust)가 1순위, B는 public 참고용.

현재 제출 폴더는 **B를 1순위로** 두고 있다. A 프로토콜의 전제가 맞다면 이 우선순위는 뒤집혀야 한다. **이 우선순위 확정 + 팀명 확정은 사용자 승인 사항**이며, git commit도 그 전까지 보류.

### 4-1. 신규 증거 (iter56): 1차 자료(primary source) 검증 — A 트랙 전제가 repo 자료와 충돌

위 두 갈래는 "별도 비라벨 최종셋이 존재하는가"에 달려 있다. 이 전제의 **유일한 출처**는
`docs/FINAL_ANTI_OVERFIT_PROTOCOL.md`의 한 줄 — *"운영위 공지에 따르면 레이블링되지 않은 최종
테스트셋이 별도로 공개될 예정"* — 이며, 이는 **2차 인용**이다. repo의 **1차 자료**를 직접 확인:

- **`docs/challenge_info.md`(공식 챌린지 요약)**: 데이터 공개는 **2차(=Validation Set, 5/4)까지뿐**.
  일정은 `Validation 예비 제출 6/1~6/5 → Validation 최종 제출 6/8 → 결과 발표 6/9`. 제출물은
  `팀이름_validation.xlsx` **하나**. **"별도 최종 테스트셋"·"데이터 공개 3차"·"final test" 언급이 전혀 없다.**
- **데이터 레이아웃(`data/raw/`, `data/convert_tdms.py`)**: `Train1~4`(라벨) + `Test1~6`(비라벨,
  "Test는 운전조건 비공개")만 존재. 즉 **Test1~6 = 공식 용어의 "Validation Set"**(코드가 Test로 명명).
- repo 전체에 "데이터 공개 3차 / 최종셋 공개 예정"의 **다른 어떤 언급도 없음**(grep 확인).

**함의**: 1차 자료대로면 **채점 대상은 Test1~6(=Validation Set) 그 자체**이고, 6/8 "최종 제출"은
*같은 Validation 예측을 확정 제출*하는 것이지 별도 held-out 셋이 아니다. 그렇다면 **A 트랙(별도
최종셋 anti-overfit)의 존재 이유가 사라지고 B(또는 증거-best 혼합)가 직접 타당**해진다. 더욱이
Test1~6은 비라벨이라 "라벨 과적합"은 애초 불가능하며, **예비 제출(6/1~6/5)의 리더보드 점수로
보정**할 수 있다.

**정직한 caveat (자율 flip 안 하는 이유)**: 프로토콜이 인용한 "운영위 공지"가 repo 밖
실제 공지(이메일/포럼)였을 **가능성을 1차 자료만으로 완전 배제할 수는 없다**. 이 한 가지가
1순위(정확도 30%)를 좌우하므로, **우선순위 확정은 여전히 사용자 승인 사항**으로 둔다.
→ **사용자 확인 필요**: "challenge_info.md 외에 별도 최종 테스트셋을 예고한 운영위 공지가 실제로 있었는가?"
  - 없었다 → A 트랙 전제 폐기, B(또는 증거-best) 1순위 확정.
  - 있었다 → 현행 헤지(B=1순위 + A=finaltest_robust 백업) 유지.

### 4-2. ★★★ 확정 (iter57): 공식 2026 챌린지 페이지(Notion)로 "별도 최종셋 없음" 결론 — A 전제 반증

사용자 제공 **공식 출처**(`grey-sedum-702.notion.site/KSPHM-KIMM-2026-…`, "제2회 KSPHM-KIMM
기계 데이터 챌린지 2026", 대회 6/24~26 부산 웨스틴조선)를 헤드리스 브라우저로 직접 렌더링·확인.
원문 그대로:

- **§4 데이터 셋 구성**: 제공 데이터는 **Train Set + Validation Set 두 가지뿐.**
  - Train Set = *"열화시험 중단 조건에 도달할 때까지 시험한 데이터로, 참가팀은 이 데이터 셋을 사용하여 모델을 학습"*
  - Validation Set = *"열화시험 중단 조건에 도달하지 못하고 베어링의 결함이 진전되고 있는 상태까지 시험한 데이터로, 참가팀은 이 데이터 셋을 사용하여 잔여수명을 예측하고 그 결과를 제출"* ← **이것이 Test1~6**.
  - **별도/비공개/최종 테스트셋 언급 전무.**
- **주요 일정 표**: `Validation 예비 제출 6/1~6/5 · Validation 최종 제출 6/8 · 결과 발표 6/9 · 발표 평가 6/25`.
- **§6 제출**: 예비(6/1~5)=`팀이름_validation.xlsx`(RUL Score) 업로드 / 최종(6/8)=validation.xlsx+code.zip+report.pdf 3종.
- **유의사항(평가 대상)**: *"본 챌린지는 **검증용 데이터 셋에 대한 예측 결과**를 기준으로 성능을 평가합니다."*
  - *"예비 제출은 모델의 성능을 사전 확인하기 위한 용도이며, 제출된 결과는 **중간 순위로 공개**됩니다."*
  - *"예비 제출 결과는 최종 평가에는 반영되지 않으며, **최종 순위는 6월 8일 최종 제출 결과를 기준으로 평가**됩니다."*
- **2단계 심사(신규 확인)**: 예선(~6/8) = 3기준(창의성·합리성·예측정확도) → **상위 7팀 본선 진출** /
  본선(6/25) = 5기준(예선3 + 우수성·논리성) → 7팀 등수 결정.
- **공지(260527)**: *"결과 제출 링크가 공개되었습니다. 6월 1일 이전 제출 파일은 채점 대상 제외·삭제."*
  → 사용자가 기억한 "추후 공개한다"는 **결과 제출 링크 공개**를 가리킨 것으로 보임(별도 테스트셋 아님).

**결론**: A 트랙의 전제("별도 비라벨 최종셋")는 **공식 출처로 반증됨**. 채점 대상 = **Validation Set
= Test1~6**, 최종 순위 = 6/8 제출(같은 셋). 따라서 **A(finaltest_robust)는 1순위 후보에서 내려가고,
B(또는 증거-best per-bearing)가 정당한 1순위**. `26_FinalRobust`는 "별도셋 대비 robust" 근거를
잃으므로 **진단/대안 기록용으로만** 보존(주력 트랙 아님).

**전략적 함의(중요)**: ① 채점 대상이 곧 Test1~6이고 **예비 제출(6/1~5)에 중간 순위(실측 점수
피드백)가 공개**되므로, 블라인드로 B vs A를 고민할 이유가 사라짐 — **예비 창에서 최강 후보를 올려
실측 점수를 받고, 6/8 최종을 그 신호로 확정**하는 것이 지배 전략. ② 예선이 정확도뿐 아니라
창의성·합리성도 보므로(본선 진출 게이트), report.pdf·창의성 narrative는 6/8 이전에 완비 필요.
**(1순위 파일 재지정/26 deprecation은 사용자 최종 확인 후 실행 — 본 절은 검증 사실 기록.)**

## 6. 신규 증거 (iter42): 동일 EOL 끝점 공정 비교 — A의 anchor near-EOL 약점

두 트랙 LOBO는 서로 다른 지점을 평가해 직접 비교 불가였다(A=EOL 끝점만, B=frac0.25~0.9만).
B의 선택 철학을 **A가 평가한 것과 동일한 held-out EOL 끝점(true=600s)**에서 돌려 공정 비교
(`experiments/.../eol_headtohead.py` → `29_eol_headtohead.csv`, train-only·누수無):

| @ true EOL 600s | pred | asym_score (4-fold 평균) |
|---|---|---|
| A frozen-anchor (5_HIBlend) | 6,000~16,800s | **0.0000** |
| B asym-optimal | ~1,200s | **0.0319** |
| B q25 | ~1,200s | 0.0313 |
| B median | 2,400s | 0.0000 |

- **정직한 해석**: 600s는 측정 하한이라 2.5× late 페널티가 모두를 0 근처로 짓누른다(B 원 LOBO가 frac0.9에서 멈춘 이유). **그러나** B 선택은 ~1,200s(자릿수 더 근접), A anchor는 6,000~16,800s. A는 EOL gate가 떠야 살지만 **held-out에서 gate는 4 fold 중 1번만 발화**(`26_..._lobo.csv` reason 열).
- **함의**: near-EOL 베어링(Test5 HI=0.944, 가능성 Test6)에서 B의 per-bearing 선택이 late 페널티에 구조적으로 더 안전. A는 신뢰도 낮은 gate에 의존 — Test5=644가 맞은 건 gate가 거기서 발화했기 때문(우연성 내포).
- **단, 전부는 아님**: 이는 한 차원(near-EOL)일 뿐. A의 설계 강점은 비라벨 최종셋에서 public 수동튜닝 회피(robustness). 이 증거는 "A의 anchor가 EOL에 약하니 gate 신뢰도 보강 필요" 또는 "near-EOL은 B가 우위"로 읽되, mid-life regime 우열은 별개.

## 7. 신규 증거 (iter43): mid-life HI-band prior 공정 비교 — 반대 방향 우열 = anti-correlation 정량 확인

§6은 '600s 끝점' 관점(B 유리). 그러나 두 트랙 최대 충돌은 mid-life 4 베어링(Test1/2/4/6 ~3×, Test3 7.6×).
반대 관점인 **HI-band prior**("동일 HI면 동일 RUL 분포")로 공정 비교: 각 Test의 last-HI에서 train 전체
HI-KNN(K=20) 이웃의 실제 rul_s 분포를 잡아, 두 트랙의 **제출값**에 대한 E[asym_score]을 계산
(`experiments/.../midlife_headtohead.py` → `30_midlife_headtohead.csv`, train-only).
비순환성: B mid-life는 28_EOLReg, A는 5_HIBlend anchor — 둘 다 이 KNN 분포의 argmax가 아님 → 공정.

| Bearing | HI | train RUL median | B pred → E[A] | A pred → E[A] | 우위 |
|---|---|---|---|---|---|
| Test1 | 0.46 | 36,900 | 10,067 → 0.37 | 32,035 → **0.71** | A |
| Test2 | 0.50 | 30,900 | 10,998 → 0.43 | 33,556 → **0.60** | A |
| Test3 | 0.165 | **72,900** | 48,900 → **0.71** | 6,449 → 0.29 | B |
| Test4 | 0.45 | 37,500 | 9,545 → 0.37 | 14,113 → **0.44** | A |
| Test5 | 0.94 | 3,600 | 644 → 0.36 | 644 → 0.36 | tie |
| Test6 | 0.41 | 47,400 | 10,275 → **0.34** | 3,000 → 0.27 | B |

- **mid-life 4 평균: A=0.508 vs B=0.378.** §6의 600s-LOBO(B=0.519>naive)와 **정반대 방향** → insight#2
  (LOBO ↔ Sensitivity anti-correlation) **정량 확인**. 두 트랙은 *서로 다른 prior에 건 베팅*:
  - **B** = "test 베어링은 HI가 시사하는 것보다 빨리 죽는다"(600s-LOBO·비대칭 보수). HI가 transfer 안 하면 우위.
  - **A** = "HI가 베어링 간 transfer 한다(HI0.46→~30000s)". HI-RUL 매핑이 일반화하면 우위.
- **red flag 2개**:
  1. **A의 Test3=6,449 위험**: 저-HI(0.165)인데 train 동일-HI는 ~73,000s. HI-prior서 0.29(B 0.71). A anchor가 저HI에서 과도하게 짧음.
  2. **B의 Test1/2/4(~10,000s) 공격적-짧음**: HI transfer 시 베어링당 0.25~0.34 손실.
- **결론(핵심 의사결정 근거)**: 트랙 선택 = "HI가 transfer 하는가"에 대한 베팅. 어느 관점도 천장(E*≈0.68~0.75)엔 못 미침.
  **'최적 하이브리드'(A mid-life + B Test3) 생성은 금지** — prior를 미리 확정해야 가능하고, 프로토콜이 경고한 과적합.
  대신 현 패키지가 두 베팅을 모두 보유(B=1순위, A=finaltest_robust)하는 헤지가 정합적. 채점 대상 확정 시 1순위 선택.

## 8. 신규 증거 (iter45): HI→RUL 전이력 정량화 — "HI가 transfer 하는가?" 직접 측정

§7까지는 per-point E[A] 비교였고, 트랙 선택의 crux("HI가 베어링 간 transfer 하는가")를 직접
측정한 적은 없었다. train-only 진단(`hi_transfer_diagnostic.py` → `31_hi_transfer_diagnostic.csv`):

**(1) 동일-HI RUL scatter** (같은 HI에서 베어링 간 RUL 변동):

| HI 구간 | RUL median | RUL min~max | CV |
|---|---|---|---|
| 0.30–0.40 | 53,700 | 36,600~71,400 | 0.17 |
| 0.40–0.50 | 39,600 | 17,400~59,400 | **0.24** |
| 0.50–0.60 | 31,800 | 15,600~49,800 | **0.32** |

→ HI는 신호를 담지만 scatter 상당(HI~0.45서 RUL 3.4× 스프레드). HI가 높아질수록 불확실 증가.

**(2) LOBO HI-only 전이** (3베어링 log(rul)~HI fit → held-out 전 측정점 예측):

| held-out | MAPE | asym | R²(log) |
|---|---|---|---|
| Train1 | 35% | 0.726 | 0.82 |
| Train2 | 38% | 0.685 | 0.80 |
| Train3 | 82% | 0.242 | 0.54 |
| Train4 | 48% | 0.554 | 0.45 |
| **평균** | **50%** | **0.552** | **0.65** |

**핵심 함의**:
- HI 단독 선형 fit이 out-of-bearing asym **0.552** 달성 → **A의 'HI transfer' mid-life 베팅은 임의가 아니라 데이터로 정당화됨** (HI가 typical 베어링엔 실제 전이 신호 보유). per-bearing 선택(0.519)·A mid-life(0.508)와 동급 이상(단 프로토콜 상이, 직접 순위 아님).
- **그러나 베어링 의존적**: Train1/2 전이 우수(R²0.8+), Train3 붕괴(R²0.54, asym0.24). 즉 atypical 베어링엔 HI 전이 실패.
- **종합**: HI 전이는 real-but-imperfect → (a) A의 mid-life 긴 예측 defensible, (b) Train3類 atypical 위험 + 동일-HI scatter는 irreducible 불확실성 → **헤지(양 트랙 보유) 정합**. 4 베어링 한계의 정직한 정량화 = 합리성 근거.

## 9. 신규 방법론 (iter46): 열화속도 기반 물리 RUL — 독립 3rd-party disambiguation

두 트랙은 'HI-level→RUL'에 의존(전이 R²=0.65). **독립 물리 모델**로 교차검증:
`RUL = elapsed × (1−HI)/HI` (HI를 life-start부터 일정 평균율 가정; 고장 HI≈1.0, train 4개서 확인).
파라미터 거의 無·train-based·임의clamp無. `experiments/32_DegradationRate_RUL/predict.py`.

- **검증**: train progression 16점 인과 평가 asym **0.586** — 지금까지 단순 train 방법 중 **최고**
  (B per-bearing 0.519 / A HI-prior 0.508 / HI-only fit 0.552 모두 상회).
  (trailing-국소-slope 변형은 HI 노이즈로 0.374 실패 — 정직한 음성 결과로 기록.)
- **Test 독립 판정** (avg-rate, elapsed=29,400s):

| Test | HI | avg-rate RUL | 함의 |
|---|---|---|---|
| Test5 | 0.94 | **1,310** | SHORT → 두 트랙 644를 **3번째 독립 방법이 확증** |
| Test1/2/4 | ~0.46 | 19k–36k | LONG → mid-life **A(긴)** 지지 (B의 ~10k보다) |
| Test3 | 0.165 | **36,626** | LONG → B의 48,900 지지. **A의 6,449는 2개 독립 방법(HI-prior+열화율)이 반박** |
| Test6 | 0.41 | 48,506 | LONG → **두 트랙 모두 Test6 과소예측** 가능성 시사 |

- **핵심 함의**: 물리 모델(최고 LOBO)이 (1) Test5 짧음 확증, (2) mid-life 긴 RUL=A 지지, (3) **A의 Test3=6,449를 명백한 오류로 지목**. 단 avg-rate는 HI 가속(EOL 근처 convex)을 무시해 체계적 과대예측 경향 → raw 제출보다는 **disambiguation 증거**로 사용. 1순위 변경은 사용자 결정 사항(트랙 소유권).

## 10. 신규 (iter48): 2축 건강평가(HI × energy-severity) + iter46/47 Test6 자체정정

iter46/47 HI-단독 모델이 Test6를 'long'으로 오판 → 다축 점검. **HI=진행도, energy_ratio·rms_multi
=심각도(severity)** 로 분리됨. severity 임계 = train near-EOL(rul≤3000)의 p90 (train 유래, 임의값 無).
`experiments/32_DegradationRate_RUL/severity_two_axis.py` → `34_severity_two_axis.csv`.

| Test | HI | energy | rms | regime |
|---|---|---|---|---|
| Test1/2/4 | 0.45~0.50 | 2.7~5.5 (정상) | 0.18~0.28 (정상) | **mid-life → 긴 RUL** |
| Test3 | 0.165 | 3.7 (정상) | 0.22 (정상) | **early → 매우 긴 RUL** |
| Test5 | 0.944 | 10.9 (>EOL p90) | 0.40 | EOL → 짧음 ✓ |
| Test6 | 0.41 | **23.3 (2× EOL p90)** | **0.61 (>EOL p90)** | **hidden severe EOL → 짧음** |

- **자체정정**: iter46/47의 'Test6=long, 두 트랙 과소예측' 주장은 **틀림**. Test6의 energy 23.3은 train EOL p90(10.8)의 2배 → 숨은 급성 열화. **A의 EOL gate 발화(→3000)는 데이터로 정당**. HI-단독 모델이 energy를 못 본 artifact였음. (정직한 음성/정정 = 합리성.)
- **샤프해진 정확도 신호**: Test1/2/4는 energy 정상+mid HI → **진짜 mid-life=긴 RUL**. 즉 **B의 ~10k가 실제 과소예측 지점**(A의 ~32k가 정합). Test6의 짧은 예측은 양 트랙 다 OK.
- **다증거 종합 per-bearing best**(4 물리/통계 방법 + 2축 severity 일치): T1/2/4≈긴(A형), T3≈긴(B의 48900), T5=644, T6=짧음(A 3000/B 10275 둘 다 가능). → 순수 A도 순수 B도 아닌 'mid-life는 길게, Test3는 길게, severe는 짧게'가 증거상 최적. 단 이를 제출본으로 조립하는 것은 test-feature 기반 per-bearing 선택 = 프로토콜이 경고한 overfit 위험 → **증거로 제시, 제출 변경은 사용자 결정.**
- **iter50 최소-보정 후보 추가**: 전체 hybrid 대신 overfit 표면을 최소화한 단일 교정만 적용 → `artifacts/submissions/HUFS_validation_finaltest_T3fix.xlsx` = finaltest_robust(A) 그대로 + **Test3만 6449→48900**(A의 유일한 식별 오류 교정). 1순위(B) 불변·비파괴 백업. A-트랙 최종 채택 시 증거-best 옵션. (6 후보 preflight PASS.)

## 5. 양 트랙 공통 안전장치 (둘 다 준수 확인)
- 임의 clamp 금지: B는 argmax_p E[A] (train 분포 내 선택), A는 LOBO-frozen 규칙. 둘 다 600s 물리 하한 + β 곱셈 보정만 사용. Test6=3000은 **임의값이 아니라** 26의 EOL gate가 high-energy EOL로 판정해 train EOL 분포 대표값으로 산출한 결과(`26_..._debug.csv` reason 열).
- 6베어링·RUL≥600·NaN無: 5개 후보 전부 통과.
