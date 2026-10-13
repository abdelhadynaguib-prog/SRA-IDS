from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from common import FEATURES, clean_bot_iot_csv, save_json

EXPECTED = {
    "training_rows": 2_934_817,
    "training_unique": 2_927_024,
    "testing_rows": 733_705,
    "testing_unique": 733_194,
    "shared_unique": 3_291,
    "seen_rows": 3_484,
    "unseen_rows": 730_221,
    "seen_normal": 6,
    "seen_attack": 3_478,
    "unseen_normal": 101,
    "unseen_attack": 730_120,
}

COMPARATOR_KEYS = {
    "Random Forest": "rf_prediction",
    "XGBoost": "xgb_prediction",
    "LightGBM": "lgb_prediction",
    "DNN": "dnn_prediction",
    "LSTM": "lstm_prediction",
    "CNN-LSTM": "cnn_lstm_prediction",
    "Attention-CNN-LSTM": "attention_prediction",
}


def row_byte_keys(frame: pd.DataFrame) -> np.ndarray:
    arr = np.ascontiguousarray(frame[FEATURES].to_numpy(dtype=np.float32), dtype=np.float32)
    row_bytes = arr.dtype.itemsize * arr.shape[1]
    return arr.view(np.dtype((np.void, row_bytes))).reshape(-1)


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_pred = np.asarray(y_pred, dtype=np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Rows": int(len(y_true)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Errors": int(fp + fn),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "Normal_Precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "Normal_Recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "Normal_F1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "Attack_Precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Attack_Recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Attack_F1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def choose_key(archive, candidates: list[str], label: str) -> str:
    for key in candidates:
        if key in archive.files:
            return key
    raise KeyError(f"Could not find {label}; tried keys: {candidates}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce STEP 26T4 exact float32 row-byte overlap and robustness audit.")
    ap.add_argument("--training-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--comparator-npz", help="STEP-26S comparator prediction archive; enables 7 comparator subset metrics")
    ap.add_argument("--sra-npz", help="Frozen SRA prediction archive; enables SRA subset metrics and historical-mask comparison")
    ap.add_argument("--output-dir", default="reproduction_outputs/11_overlap")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    x_train, _ = clean_bot_iot_csv(args.training_csv)
    x_test, y_test_series = clean_bot_iot_csv(args.test_csv)
    y_test = y_test_series.to_numpy(dtype=np.uint8)

    train_keys = row_byte_keys(x_train)
    test_keys = row_byte_keys(x_test)
    unique_train = np.unique(train_keys)
    unique_test = np.unique(test_keys)
    shared_unique = int(np.intersect1d(unique_train, unique_test, assume_unique=True).size)
    seen = np.isin(test_keys, unique_train, assume_unique=False)
    unseen = ~seen

    audit = {
        "Training_Rows": int(len(x_train)),
        "Training_Unique_Vectors": int(len(unique_train)),
        "Testing_Rows": int(len(x_test)),
        "Testing_Unique_Vectors": int(len(unique_test)),
        "Shared_Unique_Vectors": shared_unique,
        "Seen_Test_Rows": int(seen.sum()),
        "Unseen_Test_Rows": int(unseen.sum()),
        "Seen_Rate_pct": float(100.0 * seen.mean()),
        "Unseen_Rate_pct": float(100.0 * unseen.mean()),
        "Seen_Normal": int(np.sum(seen & (y_test == 0))),
        "Seen_Attack": int(np.sum(seen & (y_test == 1))),
        "Unseen_Normal": int(np.sum(unseen & (y_test == 0))),
        "Unseen_Attack": int(np.sum(unseen & (y_test == 1))),
        "Feature_Count": 12,
        "Numeric_Representation": "float32 exact row bytes",
    }

    expected_mapping = {
        "Training_Rows": EXPECTED["training_rows"],
        "Training_Unique_Vectors": EXPECTED["training_unique"],
        "Testing_Rows": EXPECTED["testing_rows"],
        "Testing_Unique_Vectors": EXPECTED["testing_unique"],
        "Shared_Unique_Vectors": EXPECTED["shared_unique"],
        "Seen_Test_Rows": EXPECTED["seen_rows"],
        "Unseen_Test_Rows": EXPECTED["unseen_rows"],
        "Seen_Normal": EXPECTED["seen_normal"],
        "Seen_Attack": EXPECTED["seen_attack"],
        "Unseen_Normal": EXPECTED["unseen_normal"],
        "Unseen_Attack": EXPECTED["unseen_attack"],
    }
    for key, expected in expected_mapping.items():
        if audit[key] != expected:
            raise RuntimeError(f"{key}: observed {audit[key]}, expected {expected}")

    mask_path = out / "common_protocol_exact_overlap_mask.npz"
    np.savez_compressed(
        mask_path,
        common_protocol_seen_mask=seen.astype(bool),
        exact_seen=seen.astype(bool),
        exact_unseen=unseen.astype(bool),
        y_true=y_test,
    )
    pd.DataFrame([audit]).to_csv(out / "common_protocol_exact_overlap_audit.csv", index=False)

    # Optional prediction-based robustness tables exactly on the frozen mask.
    systems: dict[str, np.ndarray] = {}
    historical_seen = None

    if args.comparator_npz:
        with np.load(args.comparator_npz, allow_pickle=False) as comp:
            y_key = choose_key(comp, ["y_true"], "comparator y_true")
            if not np.array_equal(comp[y_key].astype(np.uint8), y_test):
                raise RuntimeError("Comparator archive labels do not align with Testing.csv.")
            for name, key in COMPARATOR_KEYS.items():
                if key not in comp.files:
                    raise KeyError(f"Comparator archive missing {key}")
                systems[name] = comp[key].astype(np.uint8, copy=True)

    if args.sra_npz:
        with np.load(args.sra_npz, allow_pickle=False) as sra:
            y_key = choose_key(sra, ["y_true", "y_test"], "SRA y_true")
            if not np.array_equal(sra[y_key].astype(np.uint8), y_test):
                raise RuntimeError("SRA archive labels do not align with Testing.csv.")
            pred_key = choose_key(
                sra,
                ["hybrid_prediction", "sra_prediction", "prediction", "y_pred"],
                "SRA final prediction",
            )
            systems = {"Frozen SRA-IDS": sra[pred_key].astype(np.uint8, copy=True), **systems}
            for hist_key in ["seen_exact_vector_in_train", "historical_seen_mask"]:
                if hist_key in sra.files:
                    historical_seen = sra[hist_key].astype(bool, copy=True)
                    break

    if systems:
        performance_rows = []
        error_rows = []
        for system_name, pred in systems.items():
            if len(pred) != len(y_test):
                raise RuntimeError(f"{system_name}: prediction length mismatch.")
            for subset_name, mask in [
                ("Overall", np.ones(len(y_test), dtype=bool)),
                ("Exact-Seen", seen),
                ("Exact-Unseen", unseen),
            ]:
                performance_rows.append({
                    "System": system_name,
                    "Subset": subset_name,
                    **prediction_metrics(y_test[mask], pred[mask]),
                })
                wrong = pred != y_test
                error_rows.append({
                    "System": system_name,
                    "Subset": subset_name,
                    "Rows": int(mask.sum()),
                    "Errors": int(np.sum(wrong & mask)),
                    "Normal_Errors": int(np.sum(wrong & mask & (y_test == 0))),
                    "Attack_Errors": int(np.sum(wrong & mask & (y_test == 1))),
                })
        pd.DataFrame(performance_rows).to_csv(out / "common_protocol_seen_unseen_performance.csv", index=False)
        pd.DataFrame(error_rows).to_csv(out / "common_protocol_error_localization.csv", index=False)

    if historical_seen is not None:
        if len(historical_seen) != len(seen):
            raise RuntimeError("Historical SRA overlap mask length mismatch.")
        comparison = {
            "Common_Protocol_Seen": int(seen.sum()),
            "Historical_Hybrid_Seen": int(historical_seen.sum()),
            "Both_Seen": int(np.sum(seen & historical_seen)),
            "Common_Only_Seen": int(np.sum(seen & ~historical_seen)),
            "Historical_Only_Seen": int(np.sum(~seen & historical_seen)),
            "Both_Unseen": int(np.sum(~seen & ~historical_seen)),
            "Masks_Identical": bool(np.array_equal(seen, historical_seen)),
        }
        # Frozen historical result expected in the original experiment.
        if comparison["Historical_Hybrid_Seen"] != 2768:
            raise RuntimeError(
                f"Unexpected historical Hybrid seen-row count: {comparison['Historical_Hybrid_Seen']}"
            )
        if comparison["Common_Only_Seen"] != 716 or comparison["Historical_Only_Seen"] != 0:
            raise RuntimeError(f"Unexpected common-vs-historical mask comparison: {comparison}")
        pd.DataFrame([comparison]).to_csv(out / "common_vs_historical_overlap_mask_comparison.csv", index=False)
    else:
        comparison = None

    save_json(out / "step26T4_overlap_unseen_summary.json", {
        "analysis": "Exact 12-feature float32 Train/Test overlap and seen/unseen robustness audit",
        "audit": audit,
        "historical_mask_comparison": comparison,
        "prediction_based_robustness_generated": bool(systems),
        "testing_csv_read_for_overlap_reconstruction": True,
        "model_training_performed": False,
        "model_loading_performed": False,
        "model_inference_performed": False,
        "threshold_changed": False,
        "model_selection_performed": False,
        "predictions_used": "saved frozen predictions only" if systems else "none",
    })

    print(pd.DataFrame([audit]).to_string(index=False))
    print("STEP-26T4 exact overlap reproduction complete.")


if __name__ == "__main__":
    main()
