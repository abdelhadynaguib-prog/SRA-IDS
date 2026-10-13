from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common import NUMERIC_FEATURES, clean_bot_iot_csv, save_json, sha256_file

THRESHOLD = 0.50
EXPECTED_TEST_ROWS = 733_705
EXPECTED_NORMAL = 107
EXPECTED_ATTACK = 733_598
GLOBAL_BATCH_SIZE = 8192


def transform_neural(frame: pd.DataFrame, scaler) -> np.ndarray:
    continuous = scaler.transform(frame[NUMERIC_FEATURES]).astype(np.float32, copy=False)
    proto = frame["proto"].to_numpy(dtype=np.int64)
    if not np.isin(proto, [0, 1, 2, 3, 4]).all():
        raise RuntimeError("Unexpected protocol code.")
    onehot = np.eye(5, dtype=np.float32)[proto]
    out = np.ascontiguousarray(np.concatenate([onehot, continuous], axis=1), dtype=np.float32)
    if out.shape[1] != 16 or not np.isfinite(out).all():
        raise RuntimeError("Invalid neural test representation.")
    return out


def metrics(y_true: np.ndarray, p_attack: np.ndarray) -> tuple[dict, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.uint8)
    p_attack = np.asarray(p_attack, dtype=np.float64)
    y_pred = (p_attack >= THRESHOLD).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result = {
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
        "FPR": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "FNR": float(fn / (fn + tp)) if (fn + tp) else float("nan"),
        "ROC_AUC": float(roc_auc_score(y_true, p_attack)),
        "Attack_PR_AUC": float(average_precision_score(y_true, p_attack)),
        "Normal_PR_AUC": float(average_precision_score(1 - y_true, 1.0 - p_attack)),
    }
    return result, y_pred


def require_config(model_dir: Path, stem: str, model_path: Path) -> dict:
    config_path = model_dir / f"{stem}_final_refit_configuration.json"
    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing frozen artifact/config for {stem}.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("training_rows", -1)) != 2_934_817:
        raise RuntimeError(f"{stem}: unexpected training-row count in frozen config.")
    if float(config.get("decision_threshold", -1)) != THRESHOLD:
        raise RuntimeError(f"{stem}: unexpected frozen threshold.")
    if config.get("testing_csv_accessed") is not False:
        raise RuntimeError(f"{stem}: final refit config reports test access.")
    expected_sha = config.get("model_sha256")
    if expected_sha and expected_sha != sha256_file(model_path):
        raise RuntimeError(f"{stem}: model checksum differs from final-refit configuration.")
    return config


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce STEP 26S unified one-shot official-test evaluation.")
    ap.add_argument("--testing-csv", required=True)
    ap.add_argument("--model-root", required=True, help="Output directory from script 09 (contains final_refit/)")
    ap.add_argument("--output-dir", default="reproduction_outputs/10_common_protocol_test")
    ap.add_argument(
        "--allow-nonreference-hardware", action="store_true",
        help="Allow neural evaluation without exactly 2 GPUs; timing is then not reference-hardware-faithful.",
    )
    args = ap.parse_args()

    root = Path(args.model_root)
    final_root = root / "final_refit"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    x_test, y_test = clean_bot_iot_csv(args.testing_csv)
    y_true = y_test.to_numpy(dtype=np.uint8)
    if len(x_test) != EXPECTED_TEST_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_TEST_ROWS:,} test rows, observed {len(x_test):,}.")
    if int((y_true == 0).sum()) != EXPECTED_NORMAL or int((y_true == 1).sum()) != EXPECTED_ATTACK:
        raise RuntimeError("Official test class counts do not match the frozen protocol.")

    # Frozen shared neural scaler: fitted on FULL Training.csv only.
    scaler_path = final_root / "04_dnn" / "final_full_training_robust_scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing frozen neural scaler: {scaler_path}")
    scaler_load_start = time.perf_counter()
    scaler = joblib.load(scaler_path)
    scaler_load_time = time.perf_counter() - scaler_load_start
    if int(scaler.n_features_in_) != 11:
        raise RuntimeError("Frozen final neural scaler must contain 11 continuous inputs.")
    x_neural = transform_neural(x_test, scaler)

    # Reference neural test backend used by STEP 26S.
    visible_gpus = tf.config.list_physical_devices("GPU")
    reference_hardware = len(visible_gpus) == 2
    if not reference_hardware and not args.allow_nonreference_hardware:
        raise RuntimeError(
            "Reference STEP-26S neural inference requires exactly 2 GPUs. "
            "Use --allow-nonreference-hardware only for functional reproduction."
        )
    strategy = tf.distribute.MirroredStrategy() if reference_hardware else tf.distribute.get_strategy()
    if reference_hardware and strategy.num_replicas_in_sync != 2:
        raise RuntimeError("Expected 2 synchronized replicas.")

    results: list[dict] = []
    efficiency: list[dict] = []
    arrays: dict[str, np.ndarray] = {"y_true": y_true}

    def record(model_name: str, backend: str, load_time: float, infer_time: float,
               p_attack: np.ndarray, pred_key: str, prob_key: str) -> None:
        m, pred = metrics(y_true, p_attack)
        results.append({"Model": model_name, **m})
        efficiency.append({
            "Model": model_name,
            "Backend": backend,
            "Model_Load_Time_sec": float(load_time),
            "Inference_Time_sec": float(infer_time),
            "Throughput_samples_per_sec": float(len(y_true) / infer_time),
            "Latency_us_per_sample": float(infer_time * 1e6 / len(y_true)),
        })
        arrays[pred_key] = pred.astype(np.uint8)
        arrays[prob_key] = np.asarray(p_attack, dtype=np.float32)

    # Random Forest.
    rf_dir = final_root / "01_random_forest"
    rf_path = rf_dir / "random_forest_final_full_training.joblib"
    require_config(rf_dir, "random_forest", rf_path)
    t0 = time.perf_counter(); rf = joblib.load(rf_path); load_time = time.perf_counter() - t0
    t0 = time.perf_counter(); p = rf.predict_proba(x_test)[:, 1]; infer = time.perf_counter() - t0
    record("Random Forest", "CPU / sklearn", load_time, infer, p, "rf_prediction", "rf_probability")
    del rf, p; gc.collect()

    # XGBoost.
    xgb_dir = final_root / "02_xgboost"
    xgb_path = xgb_dir / "xgboost_final_full_training.json"
    require_config(xgb_dir, "xgboost", xgb_path)
    t0 = time.perf_counter(); xm = xgb.XGBClassifier(); xm.load_model(xgb_path); load_time = time.perf_counter() - t0
    t0 = time.perf_counter(); p = xm.predict_proba(x_test)[:, 1]; infer = time.perf_counter() - t0
    record("XGBoost", "CPU / XGBoost", load_time, infer, p, "xgb_prediction", "xgb_probability")
    del xm, p; gc.collect()

    # LightGBM.
    lgb_dir = final_root / "03_lightgbm"
    lgb_path = lgb_dir / "lightgbm_final_full_training.txt"
    require_config(lgb_dir, "lightgbm", lgb_path)
    t0 = time.perf_counter(); booster = lgb.Booster(model_file=str(lgb_path)); load_time = time.perf_counter() - t0
    t0 = time.perf_counter(); p = booster.predict(x_test); infer = time.perf_counter() - t0
    record("LightGBM", "CPU / LightGBM", load_time, infer, p, "lgb_prediction", "lgb_probability")
    del booster, p; gc.collect()

    # Neural models, using the same final scaler and one frozen seed-42 final refit each.
    specs = [
        ("DNN", "04_dnn", "dnn", "dnn_final_full_training_dual_gpu.keras", "dnn_prediction", "dnn_probability"),
        ("LSTM", "05_lstm", "lstm", "lstm_final_full_training_dual_gpu.keras", "lstm_prediction", "lstm_probability"),
        ("CNN-LSTM", "06_cnn_lstm", "cnn_lstm", "cnn_lstm_final_full_training_dual_gpu.keras", "cnn_lstm_prediction", "cnn_lstm_probability"),
        ("Attention-CNN-LSTM", "07_attention_cnn_lstm", "attention_cnn_lstm", "attention_cnn_lstm_final_full_training_dual_gpu.keras", "attention_prediction", "attention_probability"),
    ]
    for model_name, subdir, stem, filename, pred_key, prob_key in specs:
        model_dir = final_root / subdir
        model_path = model_dir / filename
        config = require_config(model_dir, stem, model_path)
        if reference_hardware and config.get("reference_hardware") is not True:
            raise RuntimeError(f"{model_name}: frozen config is not marked reference-hardware.")
        tf.keras.backend.clear_session()
        with strategy.scope():
            t0 = time.perf_counter()
            model = tf.keras.models.load_model(model_path, compile=False)
            load_time = time.perf_counter() - t0
        inp = x_neural if model_name == "DNN" else x_neural[..., None]
        t0 = time.perf_counter()
        p = model.predict(inp, batch_size=GLOBAL_BATCH_SIZE, verbose=0).reshape(-1)
        infer = time.perf_counter() - t0
        backend = (
            "2 x GPU / MirroredStrategy" if reference_hardware
            else "NON-REFERENCE HARDWARE / TensorFlow"
        )
        record(model_name, backend, load_time, infer, p, pred_key, prob_key)
        del model, p
        gc.collect(); tf.keras.backend.clear_session()

    result_df = pd.DataFrame(results)
    efficiency_df = pd.DataFrame(efficiency)
    result_path = out / "unified_frozen_comparator_test_results.csv"
    efficiency_path = out / "unified_frozen_comparator_inference_efficiency.csv"
    pred_path = out / "all_frozen_comparator_test_predictions.npz"
    result_df.to_csv(result_path, index=False)
    efficiency_df.to_csv(efficiency_path, index=False)
    np.savez_compressed(pred_path, **arrays)

    save_json(out / "official_test_cleaning_audit.json", {
        "rows": int(len(x_test)),
        "normal": int((y_true == 0).sum()),
        "attack": int((y_true == 1).sum()),
        "sport_minus_one": int((x_test["sport"] == -1).sum()),
        "dport_minus_one": int((x_test["dport"] == -1).sum()),
        "feature_count": int(x_test.shape[1]),
        "representation": "12 cleaned float32 predictors",
        "neural_scaler_fit_on_test": False,
        "neural_scaler_load_time_sec": float(scaler_load_time),
    })
    save_json(out / "step26S_reproduction_manifest.json", {
        "status": "FROZEN_ONE_SHOT_TEST_REPRODUCTION",
        "models_evaluated": 7,
        "common_threshold": THRESHOLD,
        "all_comparators_full_training_refits": True,
        "reference_neural_hardware": bool(reference_hardware),
        "neural_batch_size": GLOBAL_BATCH_SIZE,
        "models_retrained_after_test": False,
        "threshold_changed_after_test": False,
        "model_selection_after_test": False,
        "predictions_saved": True,
        "probabilities_saved": True,
        "post_test_tuning_permitted": False,
        "outputs": {
            "results_sha256": sha256_file(result_path),
            "efficiency_sha256": sha256_file(efficiency_path),
            "prediction_archive_sha256": sha256_file(pred_path),
        },
    })

    print(result_df.to_string(index=False))
    print(efficiency_df.to_string(index=False))
    print("STEP-26S reproduction complete; no post-test tuning permitted.")


if __name__ == "__main__":
    main()
