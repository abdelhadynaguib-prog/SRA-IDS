from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from common import (
    clean_bot_iot_csv, create_development_split, create_strict_oof_fold_ids,
    save_json, FEATURES,
)

EXPECTED = {
    "rows": 2_934_817,
    "train_rows": 2_347_854,
    "val_rows": 586_963,
    "train_normal": 296,
    "train_attack": 2_347_558,
    "val_normal": 74,
    "val_attack": 586_889,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", required=True)
    ap.add_argument("--output-dir", default="reproduction_outputs/01_split")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    x, y = clean_bot_iot_csv(args.training_csv)
    train_idx, val_idx, val_fold, sgkf_fold_id, stats = create_development_split(x, y)
    x_train = x.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    strict_fold_id = create_strict_oof_fold_ids(x_train, y_train)

    counts = {
        "rows": len(x),
        "train_rows": len(train_idx),
        "val_rows": len(val_idx),
        "train_normal": int((y.iloc[train_idx].to_numpy() == 0).sum()),
        "train_attack": int((y.iloc[train_idx].to_numpy() == 1).sum()),
        "val_normal": int((y.iloc[val_idx].to_numpy() == 0).sum()),
        "val_attack": int((y.iloc[val_idx].to_numpy() == 1).sum()),
    }
    for k, v in EXPECTED.items():
        if counts[k] != v:
            raise RuntimeError(f"{k}: observed {counts[k]}, expected {v}")

    np.savez_compressed(
        out / "development_split_indices.npz",
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        strict_oof_fold_id=strict_fold_id.astype(np.int8),
    )
    save_json(out / "development_split_metadata.json", {
        **counts,
        "selected_stratified_group_fold": val_fold,
        "features": FEATURES,
        "random_state": 42,
        "outer_candidate_folds": stats,
        "duplicate_policy": "Exact 12-feature vectors remain entirely within one split.",
        "testing_csv_read": False,
    })
    print("Development split and strict OOF fold IDs reproduced successfully.")
    print(counts)

if __name__ == "__main__":
    main()
