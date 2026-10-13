# SRA-IDS full research-workflow reproduction

This directory extends the frozen inference/reproducibility release with the
training, development, routing, benchmark, robustness, and statistical workflow
used for the SRA-IDS study.

## Provenance and scope

The numbered scripts are **method-faithful reorganizations of the archived
experiment cells**, not byte-for-byte copies of the original Kaggle notebook.
The reorganization converts the sequential notebook workflow into explicit CLI
stages with saved intermediate artifacts. Frozen counts, thresholds, fold
choices, model configurations, and statistical guardrails are asserted wherever
they were fixed by the original experiment.

The archived Zenodo artifacts remain the authoritative record of the reported
paper results. A fresh TensorFlow rerun can be numerically different at the last
few digits because GPU/CUDA/cuDNN/TensorFlow execution is environment-dependent.

The recorded common-protocol software environment was Python 3.12.13, NumPy
2.0.2, pandas 2.3.3, scikit-learn 1.6.1, LightGBM 4.6.0, XGBoost 3.2.0, and
TensorFlow 2.20.0. `requirements-full.txt` pins the recorded versions available
from the experiment log.

## Required source data

Obtain the original BoT-IoT 5% 10-best-feature files from their public source:

- `UNSW_2018_IoT_Botnet_Final_10_best_Training.csv`
- `UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv`

The dataset is not redistributed in this repository.

## Execution order

1. `01_prepare_development_split.py`
2. `02_primary_strict_nested_oof.py`
3. `03_oof_context_signals.py`
4. `04_select_router_build_verifier.py`
5. `05_verifier_strict_nested_oof.py`
6. `06_select_verifier_threshold.py`
7. `07_train_final_sra.py`
8. `08_run_official_test.py`
9. `09_train_common_protocol_comparators.py`
10. `10_evaluate_common_protocol_comparators.py`
11. `11_exact_overlap_audit.py`
12. `12_paired_statistics.py`

Scripts 01--07 use only `Training.csv`. The official test file must remain
unopened until all SRA-IDS choices are frozen. The common-protocol comparator
workflow similarly completes development selection and full-training refits
before script 10 opens `Testing.csv` once for the frozen evaluation.

## Frozen methodological checkpoints reproduced

- all 12 cleaned predictors cast to the original `float32` representation;
- leakage-safe 2,347,854 / 586,963 development split with exact-vector
  duplicate isolation;
- exact strict-OOF fold construction;
- nested Primary-LightGBM tree counts `125, 209, 113, 249, 114`, with the
  final 125 trees derived as their median rather than assumed independently;
- chunked strict-OOF Normal-leaf match computation;
- fold-specific `StandardScaler`, Normal-only reference pool, Euclidean
  `ball_tree` 3-NN search, NN1 and mean-of-three NN distance;
- router selected by minimum workload at >=99.5% Normal capture:
  `p_attack < 0.50 OR NN1 <= 1.145083189010620`;
- full 1,139-row verifier development set and duplicate/fold-isolation audit;
- strict nested-OOF residual-attention verifier with the final 29 epochs derived
  as the median of the outer-fold selections;
- final verifier threshold `0.40`;
- 20-dimensional verifier representation and 18,945 trainable parameters;
- common-protocol RF, XGBoost, LightGBM, DNN, LSTM, CNN-LSTM, and
  Attention-CNN-LSTM development/final-refit protocol;
- primary neural seed 42 with frozen final epochs 12 / 10 / 2 / 8;
- full-training neural `RobustScaler(25,75)` fitted only on Training.csv;
- reference final neural backend: two GPUs with TensorFlow
  `MirroredStrategy`, global batch 8192, float32, no final early stopping;
- one-shot seven-comparator official-test evaluation at threshold 0.50;
- exact 12-feature `float32` row-byte Train/Test overlap reconstruction;
- high-precision exact two-sided McNemar probabilities using integer
  combinations + `Decimal`, with Holm-Bonferroni applied separately to the
  seven full-test and seven exact-unseen comparisons.

## Reference-hardware note for comparator scripts

Scripts 09 and 10 enforce the original final neural reference condition of two
visible GPUs by default. `--allow-nonreference-hardware` is provided only for a
functional rerun on other hardware. A run using that flag must not be described
as reference-hardware timing reproduction.

## Typical command sequence

Set convenient paths first:

```bash
TRAIN=/path/to/UNSW_2018_IoT_Botnet_Final_10_best_Training.csv
TEST=/path/to/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv
```

Then execute stages in order. The main artifact dependencies are:

```bash
python reproduction/01_prepare_development_split.py \
  --training-csv "$TRAIN"

python reproduction/02_primary_strict_nested_oof.py \
  --training-csv "$TRAIN" \
  --split-npz reproduction_outputs/01_split/development_split_indices.npz

python reproduction/03_oof_context_signals.py \
  --training-csv "$TRAIN" \
  --split-npz reproduction_outputs/01_split/development_split_indices.npz \
  --primary-oof-npz reproduction_outputs/02_primary_oof/primary_strict_oof.npz

python reproduction/04_select_router_build_verifier.py \
  --training-csv "$TRAIN" \
  --split-npz reproduction_outputs/01_split/development_split_indices.npz \
  --context-npz reproduction_outputs/03_context/strict_oof_context_signals.npz

python reproduction/05_verifier_strict_nested_oof.py \
  --training-csv "$TRAIN" \
  --split-npz reproduction_outputs/01_split/development_split_indices.npz \
  --context-npz reproduction_outputs/03_context/strict_oof_context_signals.npz \
  --verifier-index-npz reproduction_outputs/04_router/verifier_development_index.npz

python reproduction/06_select_verifier_threshold.py \
  --training-csv "$TRAIN" \
  --split-npz reproduction_outputs/01_split/development_split_indices.npz \
  --primary-oof-npz reproduction_outputs/02_primary_oof/primary_strict_oof.npz \
  --verifier-oof-npz reproduction_outputs/05_verifier_oof/verifier_strict_oof.npz

python reproduction/07_train_final_sra.py \
  --training-csv "$TRAIN" \
  --split-npz reproduction_outputs/01_split/development_split_indices.npz \
  --primary-oof-npz reproduction_outputs/02_primary_oof/primary_strict_oof.npz \
  --context-npz reproduction_outputs/03_context/strict_oof_context_signals.npz \
  --verifier-index-npz reproduction_outputs/04_router/verifier_development_index.npz \
  --verifier-oof-npz reproduction_outputs/05_verifier_oof/verifier_strict_oof.npz \
  --router-json reproduction_outputs/04_router/frozen_router_selection.json \
  --verifier-threshold-json reproduction_outputs/06_threshold/frozen_verifier_threshold.json
```

`08_run_official_test.py` is an orchestration wrapper around the frozen
`scripts/run_frozen_test.py` utility already supplied in the base SRA-IDS
repository. Run it only after stage 07 has completed and the corresponding
frozen inference artifacts are in the locations expected by that utility.

For the controlled comparator benchmark:

```bash
python reproduction/09_train_common_protocol_comparators.py \
  --training-csv "$TRAIN" \
  --split-npz reproduction_outputs/01_split/development_split_indices.npz

python reproduction/10_evaluate_common_protocol_comparators.py \
  --testing-csv "$TEST" \
  --model-root reproduction_outputs/09_common_protocol
```

The final robustness/statistics stages accept the frozen SRA prediction archive
created by the base repository's official-test utility and the comparator NPZ
created by script 10:

```bash
python reproduction/11_exact_overlap_audit.py \
  --training-csv "$TRAIN" \
  --test-csv "$TEST" \
  --sra-npz /path/to/final_test_predictions_and_signals.npz \
  --comparator-npz reproduction_outputs/10_common_protocol_test/all_frozen_comparator_test_predictions.npz

python reproduction/12_paired_statistics.py \
  --sra-npz /path/to/final_test_predictions_and_signals.npz \
  --comparator-npz reproduction_outputs/10_common_protocol_test/all_frozen_comparator_test_predictions.npz \
  --overlap-npz reproduction_outputs/11_overlap/common_protocol_exact_overlap_mask.npz
```

## Interpretation guardrails

The controlled tree comparators had fewer frozen test errors than SRA-IDS in
some cases, and their paired differences versus SRA-IDS were not statistically
significant after Holm correction. Do not describe SRA-IDS as outperforming RF,
XGBoost, or LightGBM under the controlled benchmark. The statistically
significant paired differences were against the four tested neural comparators.
Timing comparisons are backend-specific and must not be interpreted as pure
architecture-speed comparisons.
