# Experiment Backlog from Ajou PPT and Literature

## Completed / Implemented

| ID | Method | Purpose | Status |
|---|---|---|---|
| 13 | EOLHazardGate_Calibrator | Prevent late-prediction blowups near EOL | completed |
| 14 | RPMAwareOrderFeatures | Correct BPFI/BPFO/BSF/FTF order features using RPM | completed |
| 15 | TrajectoryKNN_DTW_RUL | Match test degradation trajectories to train windows | completed |
| 16 | ScoreAware_CalibratedEnsemble | Generate safe/balanced/aggressive score-aware submissions | completed |

## New Literature-Driven Experiments

| ID | Method | Inspired by | What it tests | Expected impact |
|---|---|---|---|---|
| 17 | ConditionalMADA_StageAlignment | Tian 2023 MDA-LETCN, Liu 2020 UDA-DANN | stage-wise Train/Test domain discrepancy and source weights | prevent bad target alignment |
| 18 | MMD_CORAL_SourceWeighting | Ding 2022 multi-source DA, Hu 2022 Wasserstein DA | non-adversarial source weighting by MMD/CORAL | robust with tiny data |
| 19 | DomainSpecificResidual_Calibrator | DIDSR/sparse DA idea | split invariant prediction and domain-specific residual risk | reduce Test6/Test4 overprediction |
| 20 | UncertaintyWeighted_TargetAdaptation | dynamic hybrid DA, ensemble uncertainty | lower-quantile prediction when methods disagree | improve asymmetric score safety |
| 21 | VMD_CBAM_FeatureDenoising | Liu 2025 PLOS VMD + BiLSTM-CBAM | denoised envelope/order features or temporal attention | medium effort; later |
| 22 | ParallelTCNTransformer_Branch | Tang 2024 parallel TCN+Transformer | separate local/global branches | high effort; later |

## Completed Advanced Coverage

| ID | Method | Output | Decision |
|---|---|---|---|
| 21 | VMD_CBAM_FeatureDenoising | `03_김현우_Ensemble/artifacts/results/21_VMD_CBAM_FeatureDenoising/` | diagnostic only; overpredicts Test5/Test6 vs 16 |
| 22 | ParallelTCNTransformer_Branch | `03_김현우_Ensemble/artifacts/results/22_ParallelTCNTransformer_Branch/` | LOBO last unstable; diagnostic only |
| 23 | FullConditionalMADA_Trainable | `03_김현우_Ensemble/artifacts/results/23_FullConditionalMADA_Trainable/` | trainable MADA-lite tried; overpredicts Test3/Test6 |
| 24 | Wasserstein_SourceWeighted_RUL | `03_김현우_Ensemble/artifacts/results/24_Wasserstein_SourceWeighted_RUL/` | source weighting tried; overpredicts EOL risk cases |
| 25 | StageAwareTransformer_DA | `03_김현우_Ensemble/artifacts/results/25_StageAwareTransformer_DA/` | useful EOL confirmation; Test6 3794s, Test5 1842s |
| 26 | FinalRobust_LOBOFrozenSelector | `03_김현우_Ensemble/experiments/26_FinalRobust_LOBOFrozenSelector/` | final test scaffold; prevents public validation overfitting |
| 27 | EOLClassifier_LOBOCalibrated | `03_김현우_Ensemble/artifacts/results/27_EOLClassifier_LOBOCalibrated/` | Train-only EOL labels; improves final-safe EOL recall |

Comparison matrix: `03_김현우_Ensemble/artifacts/results/comparison/advanced_16_25_submission_matrix.csv`.

Current decision: none of 21-25 clearly supersedes `16_scoreaware_balanced_submission.xlsx`; `25_stageaware_transformer_da_submission.xlsx` is the strongest conservative diagnostic for Test5/Test6.

Final-test decision: do not directly tune on public Validation/Test1~6. Use `05 anchor → 26 selector → 27 EOL classifier` once final unlabeled test data is released.

## Execution Policy

- Run 17-20 as lightweight diagnostics/calibrators first, because training data has only four run-to-failure bearings.
- Do not replace 16 balanced unless a new candidate improves safety without increasing late-risk.
- Heavy deep models (21-22) are lower priority than calibration and feature correction.
