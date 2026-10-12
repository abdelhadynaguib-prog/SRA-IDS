# Reproducibility protocol

## Frozen-before-test rule

All adaptive decisions were completed before official-test evaluation:

- Primary LightGBM tree count
- class weighting
- preprocessing
- routing representation
- Normal-reference construction
- router threshold
- verifier architecture
- verifier training duration
- verifier threshold

No retraining, threshold modification, router modification, or post-test model selection is permitted when reproducing the frozen result.

## Frozen configuration

### Primary
- LightGBM
- 12 source predictors
- Normal sample weight: 400
- Attack sample weight: 1
- Trees: 125
- Decision threshold: 0.50

### Router
- 11 numeric variables standardized with `StandardScaler`
- five protocol one-hot indicators
- 16-dimensional routing representation
- Euclidean Normal-reference nearest-neighbor search
- rule: `p_attack < 0.50 OR NN1 <= 1.145083189010620`
- `NN3` is the mean distance to the three nearest Normal references

### Verifier
- input dimension: 20
- token projection: 32
- attention heads: 4
- key dimension: 8
- feed-forward: 32 -> 64 -> 32
- residual attention blocks: 2
- activation: Swish
- attention/FFN dropout: 0.10
- classification dropout: 0.15
- Adam, learning rate 1e-3
- binary cross-entropy
- batch size: 64
- final epochs: 29
- decision threshold: 0.40

## Verifier preprocessing

Signed-log transform is applied to the 11 numerical traffic predictors plus `NN1_distance` and `NN3_distance`.

`primary_p_attack` and `leaf_match_rate` remain direct bounded signals.

The resulting 15 continuous variables are passed through the frozen `RobustScaler(quantile_range=(10, 90))`, then combined with five protocol one-hot indicators.

## Exact-overlap audit

Exact overlap is defined by complete row-byte equality of the cleaned 12-feature `float32` vector.

Use the term **exact-unseen** for a test row with no exact training-vector match. This does not imply temporal novelty, approximate-distance novelty, or semantic novelty.

## Common-protocol benchmark

The seven standalone comparators use the same controlled protocol and the main common threshold of 0.50. No official-test threshold tuning is permitted.

Paired comparisons use exact two-sided McNemar tests and Holm correction, applied separately to the full-test and exact-unseen families.
