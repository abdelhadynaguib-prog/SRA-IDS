## Full research-workflow reproduction

The `reproduction/` directory provides a method-faithful CLI reorganization of
the archived SRA-IDS experimental workflow: leakage-safe development splitting,
strict nested OOF Primary LightGBM, chunked OOF structural/neighborhood signals,
router selection, strict nested-OOF residual-attention verification, frozen
final SRA-IDS training, the controlled seven-model comparator benchmark,
exact-overlap robustness analysis, and high-precision paired statistics.

These files are not presented as byte-for-byte copies of the original Kaggle
cells. They preserve the frozen experimental protocol and expose explicit saved
intermediate artifacts for reproducibility. See
[`reproduction/README.md`](../reproduction/README.md) for provenance, hardware
requirements, execution order, and reporting guardrails.
