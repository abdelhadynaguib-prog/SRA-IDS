from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef, balanced_accuracy_score

from common import clean_bot_iot_csv, save_json

def evaluate(threshold, full_y, primary_prob, source, verifier_prob):
    final = (primary_prob >= 0.50).astype(np.uint8)
    final[source] = (verifier_prob >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(full_y, final, labels=[0,1]).ravel()
    return {
        "threshold": float(threshold),
        "TN_hybrid": int(tn), "FP_hybrid": int(fp),
        "FN_hybrid": int(fn), "TP_hybrid": int(tp),
        "hybrid_errors": int(fp+fn),
        "hybrid_MCC": float(matthews_corrcoef(full_y, final)),
        "hybrid_balanced_accuracy": float(balanced_accuracy_score(full_y, final)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", required=True)
    ap.add_argument("--split-npz", required=True)
    ap.add_argument("--primary-oof-npz", required=True)
    ap.add_argument("--verifier-oof-npz", required=True)
    ap.add_argument("--output-dir", default="reproduction_outputs/06_threshold")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    _, y = clean_bot_iot_csv(args.training_csv)
    split = np.load(args.split_npz)
    full_y = y.iloc[split["train_idx"]].reset_index(drop=True).to_numpy(dtype=np.uint8)
    primary = np.load(args.primary_oof_npz)["p_attack"].astype(np.float64)
    ver = np.load(args.verifier_oof_npz)
    vp = ver["verifier_p_attack"].astype(np.float64)
    source = ver["source_train_index"].astype(np.int64)

    exact_thresholds = np.unique(np.concatenate([
        np.unique(vp),
        [np.nextafter(vp.max(), np.inf)]
    ]))
    exact = pd.DataFrame([evaluate(t, full_y, primary, source, vp) for t in exact_thresholds])
    exact.to_csv(out / "exact_verifier_threshold_audit.csv", index=False)

    refs = np.arange(0.10, 0.91, 0.10)
    ref = pd.DataFrame([evaluate(t, full_y, primary, source, vp) for t in refs])
    ref.to_csv(out / "reference_verifier_thresholds.csv", index=False)

    row040 = ref[np.isclose(ref["threshold"], 0.40)].iloc[0]
    if int(row040["FP_hybrid"]) != 9 or int(row040["FN_hybrid"]) != 20:
        raise RuntimeError("Threshold 0.40 no longer reproduces expected OOF confusion counts.")
    if not np.isclose(float(row040["hybrid_MCC"]), 0.952059421095, atol=1e-12):
        raise RuntimeError("Threshold 0.40 no longer reproduces expected OOF MCC.")

    save_json(out / "frozen_verifier_threshold.json", {
        "threshold": 0.40,
        "rationale": (
            "0.40 reproduces the best strict-OOF hybrid confusion matrix/MCC "
            "while avoiding an unnecessarily precise score-derived threshold."
        ),
        "expected_FP": 9, "expected_FN": 20,
        "expected_MCC": 0.952059421095,
    })
    print(row040.to_dict())

if __name__ == "__main__":
    main()
