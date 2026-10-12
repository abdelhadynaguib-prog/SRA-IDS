# SRA-IDS

**SRA-IDS: A LightGBM-Guided Selective Residual-Attention Framework for Intrusion Detection in Highly Imbalanced IoT Networks**
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23323669.svg)](https://doi.org/10.5281/zenodo.23323669)
This repository contains the source code, frozen configuration, compact result tables, statistical summaries, and integrity metadata supporting the SRA-IDS study on the BoT-IoT dataset.

## Method summary

SRA-IDS uses three stages:

1. A cost-sensitive LightGBM primary detector.
2. A lightweight router that forwards a flow if:
   - `primary_p_attack < 0.50`, or
   - `NN1_distance <= 1.145083189010620`.
3. A compact residual multi-head self-attention verifier for routed observations, with threshold `0.40`.

## Frozen official-test result

On 733,705 BoT-IoT test flows:

- TN = 105
- FP = 2
- FN = 7
- TP = 733,591
- Total errors = 9
- Accuracy = 99.998773%
- Balanced Accuracy = 99.064943%
- MCC = 0.959148
- Normal F1 = 95.8904%
- Attack F1 = 99.9994%
- Verifier routing rate = 0.053291%

## Repository structure

```text
SRA-IDS/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── .gitignore
├── GITHUB_UPLOAD_CHECKLIST.md
├── src/sra_ids/
├── scripts/
├── configs/
├── results/
├── comparator_results/
├── manifests/
├── docs/
└── artifacts/frozen/
```

## Dataset

The BoT-IoT dataset is not redistributed here. See `docs/DATASET.md`.

Expected files:

- `UNSW_2018_IoT_Botnet_Final_10_best_Training.csv`
- `UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv`

The frozen source representation uses 12 predictors:

`proto`, `sport`, `dport`, `stddev`, `N_IN_Conn_P_SrcIP`, `min`,
`state_number`, `mean`, `N_IN_Conn_P_DstIP`, `drate`, `srate`, `max`.

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

The exact package versions used by the frozen Kaggle run were exported in the original `environment.json` artifact. That file will be included in the permanent Zenodo archive. This GitHub package intentionally avoids inventing version pins that were not recovered verbatim.

## Frozen binary artifacts

Large/binary trained artifacts are intentionally excluded from normal Git history and will be archived on Zenodo.

After downloading the archive, place the files under:

```text
artifacts/frozen/
```

Expected filenames and SHA-256 digests are listed in:

```text
manifests/frozen_artifact_sha256_expected.csv
```

Verify them with:

```bash
python scripts/verify_frozen_artifacts.py   --artifact-dir artifacts/frozen   --manifest manifests/frozen_artifact_sha256_expected.csv
```

## Reproduce frozen official-test inference

```bash
python scripts/run_frozen_test.py   --test-csv /path/to/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv   --artifact-dir artifacts/frozen   --output-dir reproduced_test
```

The script reproduces:

- deterministic protocol and port preprocessing;
- Primary LightGBM probabilities;
- 16-dimensional routing representation;
- NN1 and mean-3-NN Normal-reference distances;
- Normal leaf-match rate;
- frozen OR routing rule;
- 20-dimensional verifier transformation;
- selective residual-attention inference;
- final classification metrics when the `attack` column is present.

## Frozen configuration

Primary:

- Normal-class weight = 400
- Attack-class weight = 1
- Trees = 125
- Threshold = 0.50

Router:

- Probability gate: `p_attack < 0.50`
- NN1 rescue: `NN1 <= 1.145083189010620`

Verifier:

- Input dimension = 20
- Parameters = 18,945
- Heads = 4
- Key dimension = 8
- Residual attention blocks = 2
- FFN = 32 -> 64 -> 32
- Swish activation
- Dropout = 0.10; classifier dropout = 0.15
- Adam, learning rate = 1e-3
- Batch size = 64
- Frozen epochs = 29
- Threshold = 0.40

See `configs/model_hyperparameters.json`.

## Common-protocol comparator evaluation

The benchmark contains:

- Random Forest
- XGBoost
- LightGBM
- DNN
- LSTM
- CNN-LSTM
- Attention-CNN-LSTM

The main common-protocol threshold is 0.50, with no official-test threshold tuning.

High-precision paired McNemar/Holm results are under `comparator_results/`.

The correct interpretation is:

- Random Forest, XGBoost, and LightGBM do not differ significantly from SRA-IDS after Holm correction.
- All four neural comparators differ significantly from SRA-IDS after Holm correction.
- Non-significance is not evidence of equivalence.

## Exact-overlap robustness

Exact overlap is based on exact row-byte equality of the cleaned 12-feature `float32` vectors:

- Exact-seen: 3,484 rows (0.474850%)
- Exact-unseen: 730,221 rows (99.525150%)
- All 9 SRA-IDS errors occur in the exact-unseen subset.

The documentation therefore uses **exact-unseen**, not the broader phrase "previously unseen".

## Reproducibility rule

Do not use the official test set to change:

- model parameters,
- preprocessing,
- router,
- thresholds,
- verifier epochs,
- model selection.

The frozen study used the official test set only after all adaptive decisions had been fixed.

## License

The repository code is released under the MIT License.

The BoT-IoT dataset is not covered by this license and remains subject to the original provider's terms.

## Citation

See `CITATION.cff`.

After Zenodo deposition, add the permanent DOI to this README and to `CITATION.cff`.

## Contact

Please use the corresponding-author contact listed in the manuscript.
