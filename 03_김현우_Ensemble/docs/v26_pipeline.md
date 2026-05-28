# 7_DomainAdv_Dynamics_TFT Pipeline (legacy v26)

## 위치 (압축 ID)

- 공식 이름: **7_DomainAdv_Dynamics_TFT**
- Legacy ID: **v26**
- 베이스: **6_Dynamics_DTW_TFTBiLSTM** (= legacy v25, dynamics features)
- 추가: Domain Adversarial Training (DAT) + HI-stage CE

## 모티베이션

LOBO는 베어링 단위 평가이지만, Train1~4와 Validation Test1~6 사이의 **feature 분포 자체가 다를 수 있다**. 베어링마다 결함 모드 / 채널 감도 / 노이즈 / 운전조건이 다르고, Val/Test의 운전조건 CSV는 비어있어 진동 피처에만 의존한다.

이를 보정하기 위해 Wen et al. 2021 (DANN), Tian et al. 2023 (MDA-LETCN), Ding et al. 2022 (multi-source DA) 계열에서 영감을 얻어 다음을 결합한다.

## 구조

```
        ┌─── RUL head ──────────────► RUL prediction
        │
Backbone ├─ GradReverse(λ_da)   ─ DomainHead     ─► bearing classification
  (v25)  │   (8 classes: Train1..4, Test1..6)
        │
        └─ GradReverse(λ_stage) ─ StageHead      ─► degradation stage
                                  (early/mid/late, RUL 비율 기준)
```

- **Backbone**: TFT (TCN-Transformer), v25와 동일 dm=64
- **GradReverse**: forward는 identity, backward는 -λ
- **λ schedule**: warm-up sigmoid `λ = λ_max · (2/(1+exp(-10p)) - 1)`, p = ep/EPOCHS
- λ_da_max = 0.10, λ_stage_max = 0.05

## 학습 절차 (단일 fold)

1. Train ∈ {3 bearings}, Val ∈ {1 bearing} (LOBO)
2. Source mini-batch: train_seq (X_src, y_src, dom_src, stage_src)
3. Target mini-batch: test_seq (X_tgt, dom_tgt) — y 없음
4. Forward(X_src) → RUL loss + Dom CE(src) + Stage CE
5. Forward(X_tgt) → Dom CE(tgt) (target도 grad-reverse 통과)
6. Backward로 backbone은 도메인 분류를 **속이도록** 갱신 → invariant feature

## Loss

```
total = α · weighted_asym_MSE + (1-α) · score_loss   ← RUL 회귀
      + 0.5 · (CE_dom_src + CE_dom_tgt)              ← domain adv
      + CE_stage                                     ← stage-aware
```

## Pseudo Label

- **Domain ID**: bearing 이름 그대로 8-class (Train1..4, Test1..6)
- **Stage**: 각 베어링의 max RUL 대비 비율
  - r > 0.66 → 0 (early)
  - 0.33 < r ≤ 0.66 → 1 (mid)
  - r ≤ 0.33 → 2 (late)
- Test는 RUL을 모르므로 stage=1(mid)로 임시 부여 (분류기는 grad-reverse 통과 시 source 라벨 적합에만 기여)

## 모델 / 앙상블

- TFT 3 seeds (BiLSTM은 도메인 헤드 공유가 어려워 제외)
- SEQ_LEN = 10, EPOCHS = 200, PATIENCE = 80, MIN_EPOCHS = 60

## 산출물

- `artifacts/results/07_DomainAdv_Dynamics_TFT/lobo_v26.csv` — LOBO 점수
- `artifacts/models/07_DomainAdv_Dynamics_TFT/fold_Train*/` — fold별 모델

## 가설

- v6보다 LOBO Last는 비슷하거나 약간 낮을 수 있음(adversarial 정규화로 RUL fit이 조금 손해)
- 그러나 Test 분포에 더 가까운 invariant feature → Test 예측의 **신뢰성** 상승
- 특히 Test5 같은 outlier 케이스에서 v22 (= v4) 단일 모델의 비정상 600s 예측이 완화될 가능성
