# SRA-IDS v1.1.0 — Second source-code audit

## Audit conclusion

The previously identified source-level discrepancies in the generated full-workflow
package have been corrected. The package is now suitable to publish as a
**method-faithful, reorganized reproduction workflow** alongside the frozen v1.0.0
artifacts.

It must **not** be described as a byte-for-byte export of the original Kaggle
notebook cells. The numbered scripts deliberately reorganize the archived experiment
into explicit command-line stages and saved intermediate artifacts.

## Important validation boundary

This second audit is a **source-level/static reproducibility audit** against the
archived experiment code and frozen outputs supplied for the project. It includes
syntax/AST checks and a known-case validation of the arbitrary-precision McNemar
implementation. The complete multi-million-row Training/Test experiment was not
rerun in this audit environment because the original BoT-IoT CSV files and the
reference TensorFlow dual-GPU runtime are not available here. The published Zenodo
models, predictions, and manifests therefore remain the authoritative numerical
record.

## Resolution of the seven previous findings

| Previous finding | Resolution in corrected v1.1.0 | Status |
|---|---|---|
| Shared cleaning did not freeze all 12 predictors to the original float32 representation | `common.py` now casts the complete ordered 12-feature matrix to `float32` after deterministic protocol/port cleaning | PASS |
| OOF leaf/NN context code materialized large outer-fold objects rather than using original chunking | `03_oof_context_signals.py` now uses 50,000-row chunking for both leaf prediction and Normal-reference 3-NN queries | PASS |
| Router script did not reconstruct complete STEP 15/16 behavior | `04_select_router_build_verifier.py` now evaluates exact LEAF/NN1/NN3 rescue breakpoints, applies original workload/tie ordering, asserts the frozen 1,139 / 295 / 844 routing result, builds the full 16-column verifier dataset, and audits duplicate/fold isolation | PASS |
| Final SRA training hard-coded 125 trees and 29 epochs instead of deriving them | `07_train_final_sra.py` derives 125 as the median of strict Primary OOF selected iterations and 29 as the median of strict verifier OOF selected epochs, then safety-checks the frozen values | PASS |
| Common-protocol comparator training did not reproduce the full frozen protocol | `09_train_common_protocol_comparators.py` now verifies STEP-26A index SHA-256 fingerprints, runs the five-seed development protocol, freezes seed-42 epochs 12/10/2/8, refits tree models on all Training.csv rows, fits the final 25--75 RobustScaler on full training only, and enforces the reference two-GPU MirroredStrategy final neural refit by default | PASS |
| Unified comparator test outputs were incomplete/incompatible with downstream statistics | `10_evaluate_common_protocol_comparators.py` now performs the one-shot seven-model frozen test evaluation, reports the complete classification metric set and backend/timing context, and writes the original downstream prediction keys (`rf_prediction`, `xgb_prediction`, `lgb_prediction`, `dnn_prediction`, `lstm_prediction`, `cnn_lstm_prediction`, `attention_prediction`) plus probabilities | PASS |
| Overlap/statistics scripts did not reproduce final STEP26T4/T5 protocol | `11_exact_overlap_audit.py` now reconstructs full float32 row-byte overlap counts and optional eight-system subset/error/historical-mask tables; `12_paired_statistics.py` now uses integer combinations + `Decimal` exact two-sided McNemar probabilities and separate Holm-Bonferroni families for Full Test and Exact-Unseen | PASS |

## File-by-file final classification

| File | Final classification | Audit note |
|---|---|---|
| `reproduction/common.py` | Method-faithful shared helper | Cleaning, exact grouping, split utilities and deterministic float32 representation are grounded in the archived workflow. |
| `01_prepare_development_split.py` | Method-faithful reorganization | Reproduces exact-vector grouping and the leakage-safe development split/fold artifacts. |
| `02_primary_strict_nested_oof.py` | Method-faithful reorganization | Preserves strict nested Primary-LightGBM OOF selection and frozen iteration sequence. |
| `03_oof_context_signals.py` | Method-faithful reorganization | Restores original chunked Normal-leaf and Normal-reference NN1/mean-3NN computation. |
| `04_select_router_build_verifier.py` | Method-faithful reorganization | Reproduces STEP15 router search and STEP16 verifier dataset/integrity checks. |
| `05_verifier_strict_nested_oof.py` | Method-faithful reorganization | Preserves nested verifier epoch selection and saves selected epochs for downstream derivation. |
| `06_select_verifier_threshold.py` | Method-faithful reorganization | Reproduces the frozen 0.40 threshold decision and OOF safety checks. |
| `07_train_final_sra.py` | Method-faithful reorganization | Final Primary/router/leaf/verifier artifacts derive their frozen choices from prior strict-OOF outputs. |
| `08_run_official_test.py` | Orchestration wrapper | Intentionally delegates to the already published base-repository frozen inference utility; it is not represented as an original notebook cell. |
| `09_train_common_protocol_comparators.py` | Method-faithful reorganization | Reproduces STEPS 26A--26Q comparator development/final-refit policy, including reference dual-GPU guardrail. |
| `10_evaluate_common_protocol_comparators.py` | Method-faithful reorganization | Reproduces STEP26S one-shot frozen comparator test evaluation and output schema. |
| `11_exact_overlap_audit.py` | Method-faithful reorganization | Reproduces STEP26T4 exact float32 row-byte overlap and robustness tables when frozen predictions are supplied. |
| `12_paired_statistics.py` | Method-faithful reorganization | Reproduces final STEP26T5 high-precision paired statistics without p-value underflow. |

**Literal/verbatim original Kaggle cells in this package: 0.**  This is intentional;
the release is a structured reproduction implementation, not a notebook-source dump.

## Automated static checks performed

- All 13 Python files parse successfully with Python AST.
- All 13 Python files pass `py_compile` syntax validation.
- Required source-fidelity markers were checked: float32 cleaning, 50k context chunking,
  `ball_tree` Euclidean 3-NN, 1,139 routed-row guardrail, saved verifier selected epochs,
  median-derived final tree/epoch counts, STEP26A split hashes, MirroredStrategy,
  frozen seed-42 epoch assertion, exact prediction-archive keys, `np.void` row-byte
  equality, and `Decimal`/integer-combination paired statistics.
- Known exact McNemar cases reproduce the archived values:
  - b=4, c=8 -> p=0.3876953125
  - b=1, c=6 -> p=0.125
  - b=4, c=7 -> p=0.548828125
  - very small DNN/LSTM/CNN-LSTM values remain non-zero in Decimal arithmetic.
- `__pycache__`, `.pyc`, and `.pyo` files are excluded from the release.

## Release wording

Recommended repository wording:

> The `reproduction/` workflow is a method-faithful reorganization of the archived
> experimental code into standalone command-line stages. Frozen Zenodo artifacts
> remain the authoritative numerical record of the reported study. Exact numerical
> identity of fresh neural-network training runs can depend on the software/GPU stack.
