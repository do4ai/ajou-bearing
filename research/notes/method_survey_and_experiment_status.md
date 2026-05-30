# Method Survey and Experiment Status

## Source Material

- Ajou PPT copied to `research/slides/KSPHM_KIMM_RUL_v24_methodology_HUFS.pptx`.
- Extracted slide text saved to `research/notes/ajou_ppt_extracted_text.md`.
- Bibliography tracker saved to `research/references.csv`.

## Downloaded Public PDFs

| File | Method |
|---|---|
| `papers/public/Jin_2025_Sensors_TCN_Transformer_Vibration_RUL.pdf` | TCN-Transformer vibration RUL |
| `papers/public/Cao_2024_AppliedSciences_Multidomain_TCN_RUL.pdf` | multi-domain feature + TCN |
| `papers/public/Tian_2023_AppliedSciences_MDA_LETCN_DomainAdaptation_RUL.pdf` | MDA-LETCN / multistage domain adaptation |
| `papers/public/Liu_Gryllias_2020_UDA_DANN_Bearing_RUL.pdf` | UDA-DANN for bearing RUL |
| `papers/public/Liu_2025_PLOS_VMD_BiLSTM_CBAM_RUL.pdf` | VMD + BiLSTM-CBAM* |
| `papers/public/Adversarial_DomainAdaptation_RUL_SciReports_2022.pdf` | adversarial DA RUL |
| `papers/public/TCN-TFT-BiLSTM_RUL_2024.pdf` | TCN/TFT/BiLSTM |
| `papers/public/PINN-Attention-RUL_arXiv2024.pdf` | physics-informed attention |
| `papers/public/DeepLearning-MotorBearing-DoubleLoss_2024.pdf` | double-loss bearing RUL |
| `papers/public/LLM-Transfer-RUL_arXiv2025.pdf` | transfer/LLM RUL reference |

## Restricted / Not Downloaded

- IEEE papers from PPT: metadata tracked, PDF restricted.
- ScienceDirect papers: metadata tracked, PDF restricted.
- PeerJ Wen 2021: public page exists but direct PDF returned 403; URL stored in `papers/restricted/Wen_2021_PeerJ_DomainAdaptation_RUL.url`.

## Experiments Added After Survey

| ID | Method | Output | Summary |
|---|---|---|---|
| 17 | ConditionalMADA_StageAlignment | `03_김현우_Ensemble/artifacts/results/17_ConditionalMADA_StageAlignment/` | Test5/Test6 pseudo-stage=eol; closest source is Train1 |
| 18 | MMD_CORAL_SourceWeighting | `03_김현우_Ensemble/artifacts/results/18_MMD_CORAL_SourceWeighting/` | source weights from non-adversarial MMD/CORAL distances |
| 19 | DomainSpecificResidual_Calibrator | `03_김현우_Ensemble/artifacts/results/19_DomainSpecificResidual_Calibrator/` | no additional downshift beyond 16 balanced |
| 20 | UncertaintyWeighted_TargetAdaptation | `03_김현우_Ensemble/artifacts/results/20_UncertaintyWeighted_TargetAdaptation/` | risk-aware quantile ensemble; similar to 16 balanced for key risky cases |
| 21 | VMD_CBAM_FeatureDenoising | `03_김현우_Ensemble/artifacts/results/21_VMD_CBAM_FeatureDenoising/` | VMD-lite/CBAM features tested; not a final candidate |
| 22 | ParallelTCNTransformer_Branch | `03_김현우_Ensemble/artifacts/results/22_ParallelTCNTransformer_Branch/` | parallel local/global surrogate tested; LOBO last unstable |
| 23 | FullConditionalMADA_Trainable | `03_김현우_Ensemble/artifacts/results/23_FullConditionalMADA_Trainable/` | trainable MADA-lite tested; too aggressive on non-EOL cases |
| 24 | Wasserstein_SourceWeighted_RUL | `03_김현우_Ensemble/artifacts/results/24_Wasserstein_SourceWeighted_RUL/` | Wasserstein source weighting tested; overpredicts EOL-risk cases |
| 25 | StageAwareTransformer_DA | `03_김현우_Ensemble/artifacts/results/25_StageAwareTransformer_DA/` | confirms lower EOL estimates for Test5/Test6 |

## Interpretation

- MADA-style diagnostics support the existing concern that Test5/Test6 are EOL-like, especially relative to Train1.
- MMD/CORAL does not overturn `16_scoreaware_balanced`; it provides source-distance evidence rather than a better direct submission.
- Domain-specific residual did not trigger additional changes, so it is not a stronger candidate than 16 balanced.
- Uncertainty weighting confirms that Test5 and Test6 should remain conservative.

## Current Best Practical Candidate

- Primary comparison candidate: `03_김현우_Ensemble/artifacts/results/16_ScoreAware_CalibratedEnsemble/16_scoreaware_balanced_submission.xlsx`.
- Conservative backup: `03_김현우_Ensemble/artifacts/results/16_ScoreAware_CalibratedEnsemble/16_scoreaware_safe_submission.xlsx`.
- New `20_uncertainty_weighted_submission.xlsx` is useful as a diagnostic but does not clearly supersede 16 balanced.
- Advanced 21-25 were also executed. The only result that strengthens the final decision is 25, which supports conservative Test5/Test6 handling; it still does not replace 16 balanced as the primary candidate.
