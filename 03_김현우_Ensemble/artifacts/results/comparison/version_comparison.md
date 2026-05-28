# Method Comparison — LOBO 4-fold

> 0.7×Full + 0.3×Last = Combined. 모든 점수는 1.0이 최고.

| Method | Full | Last | Combined | Note |
|---------|------|------|----------|------|
| 1_Baseline_1ch_TFTBiLSTM_GPR | — | — | — | cols: ['val_bearing', 'rmse_s', 'tft_mean', 'tft_median', 'tft_trim', 'tft_cons'] |
| 2_EOLDirect_4ch_WeightedRUL | 0.5582 | 0.9564 | 0.6776 |  |
| 4_ChannelSym_EOLWeighted | 0.5791 | 1.0000 | 0.7054 |  |
| 6_Dynamics_DTW_TFTBiLSTM | 0.5435 | 0.7500 | 0.6054 |  |
| 7_DomainAdv_Dynamics_TFT | — | — | — | missing in this generated file |
| 3_HIBlend_Baseline_EOLDirect | 0.6030 | 0.9780 | 0.7150 | from blend_v17v18_grid |
| 5_HIBlend_Baseline_ChannelSym combined | 0.5990 | 0.9750 | 0.7120 | from blend_v17v22_grid (current stable best) |
| 5_HIBlend_Baseline_ChannelSym full | 0.6370 | 0.3080 | 0.5380 | high full but risky last |

## 분석
- `5_HIBlend_Baseline_ChannelSym` combined가 현재 안정 best 후보.
- `6_Dynamics_DTW_TFTBiLSTM`, `7_DomainAdv_Dynamics_TFT`는 Train3 last 폭주 때문에 EOL gate 없이 제출 금지.
- 다음 우선순위는 `13_EOLHazardGate_Calibrator`와 `14_RPMAwareOrderFeatures`.
