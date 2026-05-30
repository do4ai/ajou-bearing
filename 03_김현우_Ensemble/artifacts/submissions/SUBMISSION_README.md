# Submission Files

## 1순위 (권장)

- 파일: `팀이름_validation_1순위.xlsx`
- 전략: per_bearing_best_mix (18_PerBearing_Robust)
- Sensitivity mean (HI-band prior): **0.4883**
- 베어링별 best method 선택:

  - **Test1** (HI=0.463, band=midlow): 28_eol_cons → 10067s (robust=0.507)

  - **Test2** (HI=0.500, band=midlow): 28_eol_med → 10998s (robust=0.466)

  - **Test3** (HI=0.165, band=low): 17_hybrid → 48900s (robust=0.568)

  - **Test4** (HI=0.448, band=midlow): 28_eol_med → 9545s (robust=0.492)

  - **Test5** (HI=0.944, band=high): 5_HIBlend_combined → 644s (robust=0.401)

  - **Test6** (HI=0.410, band=midlow): 28_eol_med → 10275s (robust=0.496)


## 백업 1

- 파일: `팀이름_validation_백업1.xlsx`
- 전략: 5_HIBlend_Baseline_ChannelSym combined
- LOBO Combined: **0.712** (검증된 안정 default)
- Sensitivity mean: 0.399


## 백업 2

- 파일: `팀이름_validation_백업2.xlsx`
- 전략: 19_EOLProgression_Robust (Train HI 곡선 fit + EOL bound cap)
- LOBO mean: **0.750** (Train1/2/4 perfect, Train3 fold 약점 공통)
- Sensitivity mean: 0.429


## 백업 선택 근거: 위험 분산

- 1순위 = Sensitivity (HI-band prior)
- 백업1 = LOBO (train 600s last 라벨 fit)
- 백업2 = EOL physics (HI 곡선 → progression)
- 세 가지 다른 가설로 robust 보장.
