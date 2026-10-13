from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.preprocessing import RobustScaler

from common import FEATURES, NUMERIC_FEATURES, clean_bot_iot_csv, save_json, sha256_file

# -----------------------------------------------------------------------------
# Frozen common-protocol benchmark constants from STEPS 26A-26J.
# -----------------------------------------------------------------------------
NORMAL_WEIGHT = 400.0
ATTACK_WEIGHT = 1.0
THRESHOLD = 0.50
PRIMARY_SEED = 42
DEEP_SEEDS = [42, 52, 62, 72, 82]
GLOBAL_BATCH_SIZE = 8192
MAX_DEV_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
EXPECTED_TRAIN_ROWS = 2_347_854
EXPECTED_VAL_ROWS = 586_963
EXPECTED_FULL_ROWS = 2_934_817
EXPECTED_TRAIN_SHA256 = "d6eb077d2ea1a7b097a7d613f77e16f178673ef39eed09a23556e6c052b6a432"
EXPECTED_VAL_SHA256 = "94a3a8bd36102dbc4cdb9fff684e809e5e06c3ffe87e4a55e72c59353566504c"
EXPECTED_FINAL_EPOCHS = {
    "DNN": 12,
    "LSTM": 10,
    "CNN-LSTM": 2,
    "Attention-CNN-LSTM": 8,
}
EXPECTED_PARAMS = {
    "DNN": 11_265,
    "LSTM": 19_009,
    "CNN-LSTM": 77_249,
    "Attention-CNN-LSTM": 94_017,
}

RF_PARAMS = dict(
    n_estimators=300,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
)
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=1.0,
    colsample_bytree=1.0,
    min_child_weight=1.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)
LGB_PARAMS = dict(
    n_estimators=125,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=42,
    deterministic=True,
    force_col_wise=True,
    verbosity=-1,
    n_jobs=-1,
)


def index_sha256(index: np.ndarray) -> str:
    arr = np.ascontiguousarray(index, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def transform_neural(frame: pd.DataFrame, scaler: RobustScaler) -> np.ndarray:
    continuous = scaler.transform(frame[NUMERIC_FEATURES]).astype(np.float32, copy=False)
    proto = frame["proto"].to_numpy(dtype=np.int64)
    if not np.isin(proto, [0, 1, 2, 3, 4]).all():
        raise RuntimeError("Unexpected protocol code in neural preprocessing.")
    proto_oh = np.eye(5, dtype=np.float32)[proto]
    out = np.ascontiguousarray(np.concatenate([proto_oh, continuous], axis=1), dtype=np.float32)
    if out.shape[1] != 16 or not np.isfinite(out).all():
        raise RuntimeError("Unexpected or non-finite neural input representation.")
    return out


def build_model(name: str) -> tf.keras.Model:
    if name == "DNN":
        inp = tf.keras.Input(shape=(16,), name="features")
        x = tf.keras.layers.Dense(128, activation="relu")(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.20)(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.20)(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    elif name == "LSTM":
        inp = tf.keras.Input(shape=(16, 1), name="feature_sequence")
        x = tf.keras.layers.LSTM(64)(inp)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.20)(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    elif name == "CNN-LSTM":
        inp = tf.keras.Input(shape=(16, 1), name="feature_sequence")
        x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
        x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LSTM(64)(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.20)(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    elif name == "Attention-CNN-LSTM":
        inp = tf.keras.Input(shape=(16, 1), name="feature_sequence")
        x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
        attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        x = tf.keras.layers.Add()([x, attn])
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.20)(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    else:
        raise ValueError(f"Unknown model: {name}")

    model = tf.keras.Model(inp, out, name=name.lower().replace("-", "_"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
    )
    return model


def neural_shape(x: np.ndarray, name: str) -> np.ndarray:
    return x if name == "DNN" else x[..., None]


def binary_metrics(y_true: np.ndarray, p_attack: np.ndarray, threshold: float = THRESHOLD) -> dict:
    y_true = np.asarray(y_true, dtype=np.uint8)
    p_attack = np.asarray(p_attack, dtype=np.float64)
    y_pred = (p_attack >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    normal_true = 1 - y_true
    normal_score = 1.0 - p_attack
    return {
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
        "Normal_PR_AUC": float(average_precision_score(normal_true, normal_score)),
    }


def save_tree_config_result(directory: Path, model_name: str, config: dict, train_time: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / f"{model_name}_final_refit_configuration.json", config)
    save_json(directory / f"{model_name}_final_refit_result.json", {
        "Model": model_name,
        "Training_Rows": EXPECTED_FULL_ROWS,
        "Training_Time_sec": float(train_time),
        "Decision_Threshold": THRESHOLD,
        "Testing_csv_accessed": False,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce STEPS 26A-26Q common-protocol comparator development and final refits.")
    ap.add_argument("--training-csv", required=True)
    ap.add_argument("--split-npz", required=True, help="development_split_indices.npz from script 01")
    ap.add_argument("--output-dir", default="reproduction_outputs/09_common_protocol")
    ap.add_argument(
        "--allow-nonreference-hardware",
        action="store_true",
        help="Allow final neural refits without exactly 2 GPUs. Such a run is functional, not reference-hardware-faithful.",
    )
    args = ap.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Reproducibility / numerical policy.
    tf.keras.mixed_precision.set_global_policy("float32")
    set_seed(PRIMARY_SEED)
    determinism = "ENABLED"
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as exc:  # pragma: no cover - backend dependent
        determinism = f"REQUESTED_BUT_NOT_CONFIRMED: {exc}"

    # ------------------------------------------------------------------
    # Load exactly the 12 cleaned float32 predictors and frozen STEP-26A split.
    # ------------------------------------------------------------------
    x, y = clean_bot_iot_csv(args.training_csv)
    if len(x) != EXPECTED_FULL_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_FULL_ROWS:,} training rows, observed {len(x):,}.")

    split = np.load(args.split_npz, allow_pickle=False)
    train_idx = split["train_idx"].astype(np.int64)
    val_idx = split["val_idx"].astype(np.int64)
    if len(train_idx) != EXPECTED_TRAIN_ROWS or len(val_idx) != EXPECTED_VAL_ROWS:
        raise RuntimeError("Unexpected common-protocol split sizes.")

    train_sha = index_sha256(train_idx)
    val_sha = index_sha256(val_idx)
    if train_sha != EXPECTED_TRAIN_SHA256 or val_sha != EXPECTED_VAL_SHA256:
        raise RuntimeError(
            "STEP-26A split fingerprint mismatch.\n"
            f"Train observed: {train_sha}\nVal observed: {val_sha}"
        )

    x_train = x.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    x_val = x.iloc[val_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)
    y_train_np = y_train.to_numpy(dtype=np.uint8)
    y_val_np = y_val.to_numpy(dtype=np.uint8)
    y_full_np = y.to_numpy(dtype=np.uint8)

    expected_counts = (
        int((y_train_np == 0).sum()), int((y_train_np == 1).sum()),
        int((y_val_np == 0).sum()), int((y_val_np == 1).sum()),
    )
    if expected_counts != (296, 2_347_558, 74, 586_889):
        raise RuntimeError(f"Unexpected class counts: {expected_counts}")

    w_train = np.where(y_train_np == 0, NORMAL_WEIGHT, ATTACK_WEIGHT).astype(np.float32)
    w_full = np.where(y_full_np == 0, NORMAL_WEIGHT, ATTACK_WEIGHT).astype(np.float32)

    # ------------------------------------------------------------------
    # STEPS 26C-26E: development validation for the three tree comparators.
    # ------------------------------------------------------------------
    validation_rows: list[dict] = []
    tree_specs = [
        ("Random Forest", RandomForestClassifier(**RF_PARAMS)),
        ("XGBoost", xgb.XGBClassifier(**XGB_PARAMS)),
        ("LightGBM", lgb.LGBMClassifier(**LGB_PARAMS)),
    ]
    for model_name, model in tree_specs:
        start = time.perf_counter()
        if model_name == "LightGBM":
            model.fit(x_train, y_train, sample_weight=w_train, categorical_feature=["proto"])
        else:
            model.fit(x_train, y_train, sample_weight=w_train)
        train_time = time.perf_counter() - start
        p = model.predict_proba(x_val)[:, 1]
        validation_rows.append({
            "Model": model_name, "Seed": 42, "Primary_Comparator": True,
            "Best_Validation_Epoch": np.nan, **binary_metrics(y_val_np, p),
            "Training_Time_sec": float(train_time),
        })
        del model, p
        gc.collect()

    # ------------------------------------------------------------------
    # STEPS 26F-26I: five-seed neural development stability experiment.
    # IMPORTANT: the development scaler is fit on benchmark Training only.
    # ------------------------------------------------------------------
    dev_scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    dev_scaler.fit(x_train[NUMERIC_FEATURES])
    dev_dir = root / "development_validation"
    dev_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(dev_scaler, dev_dir / "shared_development_robust_scaler.joblib")
    x_train_neural = transform_neural(x_train, dev_scaler)
    x_val_neural = transform_neural(x_val, dev_scaler)

    neural_primary_epochs: dict[str, int] = {}
    class_weight = {0: NORMAL_WEIGHT, 1: ATTACK_WEIGHT}

    for model_name in EXPECTED_FINAL_EPOCHS:
        model_dev_dir = dev_dir / model_name.lower().replace("-", "_")
        model_dev_dir.mkdir(parents=True, exist_ok=True)
        model_seed_rows = []

        for seed in DEEP_SEEDS:
            tf.keras.backend.clear_session()
            set_seed(seed)
            model = build_model(model_name)
            if model.count_params() != EXPECTED_PARAMS[model_name]:
                raise RuntimeError(
                    f"{model_name} parameter count mismatch: {model.count_params()} != {EXPECTED_PARAMS[model_name]}"
                )

            train_input = neural_shape(x_train_neural, model_name)
            val_input = neural_shape(x_val_neural, model_name)
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", mode="min", patience=EARLY_STOPPING_PATIENCE,
                    restore_best_weights=True, verbose=0,
                )
            ]
            start = time.perf_counter()
            hist = model.fit(
                train_input,
                y_train_np,
                validation_data=(val_input, y_val_np),
                epochs=MAX_DEV_EPOCHS,
                batch_size=GLOBAL_BATCH_SIZE,
                class_weight=class_weight,
                callbacks=callbacks,
                shuffle=True,
                verbose=0,
            )
            train_time = time.perf_counter() - start
            val_loss = np.asarray(hist.history["val_loss"], dtype=np.float64)
            best_epoch = int(np.argmin(val_loss) + 1)

            infer_start = time.perf_counter()
            p_val = model.predict(val_input, batch_size=GLOBAL_BATCH_SIZE, verbose=0).reshape(-1)
            infer_time = time.perf_counter() - infer_start
            metrics = binary_metrics(y_val_np, p_val)
            row = {
                "Model": model_name,
                "Seed": seed,
                "Primary_Comparator": bool(seed == PRIMARY_SEED),
                "Epochs_Ran": int(len(hist.history["loss"])),
                "Best_Validation_Epoch": best_epoch,
                "Best_Val_Loss": float(val_loss[best_epoch - 1]),
                **metrics,
                "Training_Time_sec": float(train_time),
                "Inference_Time_sec": float(infer_time),
                "Throughput_samples_per_sec": float(len(y_val_np) / infer_time),
                "Latency_us_per_sample": float(infer_time * 1e6 / len(y_val_np)),
                "Model_Parameters": int(model.count_params()),
            }
            model_seed_rows.append(row)
            validation_rows.append(row.copy())

            if seed == PRIMARY_SEED:
                neural_primary_epochs[model_name] = best_epoch
                model.save(model_dev_dir / "primary_seed42_validation_model.keras")
                np.savez_compressed(
                    model_dev_dir / "primary_seed42_validation_predictions.npz",
                    y_true=y_val_np,
                    p_attack=p_val.astype(np.float32),
                    y_pred=(p_val >= THRESHOLD).astype(np.uint8),
                )

            del model, hist, p_val
            gc.collect()
            tf.keras.backend.clear_session()

        pd.DataFrame(model_seed_rows).to_csv(model_dev_dir / "five_seed_validation_results.csv", index=False)

    if neural_primary_epochs != EXPECTED_FINAL_EPOCHS:
        raise RuntimeError(
            "Primary seed-42 validation epochs do not reproduce the frozen STEP-26J values.\n"
            f"Observed: {neural_primary_epochs}\nExpected: {EXPECTED_FINAL_EPOCHS}"
        )

    pd.DataFrame(validation_rows).to_csv(root / "common_protocol_development_validation_results.csv", index=False)
    pd.DataFrame([
        {"Model": name, "Primary_Seed": PRIMARY_SEED, "Frozen_Final_Epochs": epochs}
        for name, epochs in EXPECTED_FINAL_EPOCHS.items()
    ]).to_csv(root / "frozen_primary_seed42_epoch_selection.csv", index=False)

    # Release development matrices before final full-data refit.
    del x_train_neural, x_val_neural, dev_scaler, x_train, x_val, y_train, y_val
    gc.collect()

    # ------------------------------------------------------------------
    # STEPS 26K-26M: refit the three tree comparators on ALL Training.csv.
    # ------------------------------------------------------------------
    final_root = root / "final_refit"
    final_root.mkdir(parents=True, exist_ok=True)

    rf_dir = final_root / "01_random_forest"
    rf_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(x, y, sample_weight=w_full)
    rf_time = time.perf_counter() - start
    rf_path = rf_dir / "random_forest_final_full_training.joblib"
    joblib.dump(rf, rf_path)
    save_tree_config_result(rf_dir, "random_forest", {
        "model": "Random Forest", "training_rows": EXPECTED_FULL_ROWS,
        "features": FEATURES, "hyperparameters": RF_PARAMS,
        "class_weight": {"Normal": NORMAL_WEIGHT, "Attack": ATTACK_WEIGHT},
        "decision_threshold": THRESHOLD, "testing_csv_accessed": False,
        "model_sha256": sha256_file(rf_path),
    }, rf_time)
    del rf; gc.collect()

    xgb_dir = final_root / "02_xgboost"
    xgb_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(x, y, sample_weight=w_full)
    xgb_time = time.perf_counter() - start
    xgb_path = xgb_dir / "xgboost_final_full_training.json"
    xgb_model.save_model(xgb_path)
    save_tree_config_result(xgb_dir, "xgboost", {
        "model": "XGBoost", "training_rows": EXPECTED_FULL_ROWS,
        "features": FEATURES, "hyperparameters": XGB_PARAMS,
        "class_weight": {"Normal": NORMAL_WEIGHT, "Attack": ATTACK_WEIGHT},
        "decision_threshold": THRESHOLD, "testing_csv_accessed": False,
        "model_sha256": sha256_file(xgb_path),
    }, xgb_time)
    del xgb_model; gc.collect()

    lgb_dir = final_root / "03_lightgbm"
    lgb_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
    lgb_model.fit(x, y, sample_weight=w_full, categorical_feature=["proto"])
    lgb_time = time.perf_counter() - start
    lgb_path = lgb_dir / "lightgbm_final_full_training.txt"
    lgb_model.booster_.save_model(str(lgb_path))
    save_tree_config_result(lgb_dir, "lightgbm", {
        "model": "LightGBM", "training_rows": EXPECTED_FULL_ROWS,
        "features": FEATURES, "hyperparameters": LGB_PARAMS,
        "class_weight": {"Normal": NORMAL_WEIGHT, "Attack": ATTACK_WEIGHT},
        "decision_threshold": THRESHOLD, "testing_csv_accessed": False,
        "model_sha256": sha256_file(lgb_path),
    }, lgb_time)
    del lgb_model; gc.collect()

    # ------------------------------------------------------------------
    # STEP 26N: final neural scaler on ALL Training.csv only.
    # ------------------------------------------------------------------
    final_scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    final_scaler.fit(x[NUMERIC_FEATURES])
    dnn_dir = final_root / "04_dnn"
    dnn_dir.mkdir(parents=True, exist_ok=True)
    final_scaler_path = dnn_dir / "final_full_training_robust_scaler.joblib"
    joblib.dump(final_scaler, final_scaler_path)
    x_full_neural = transform_neural(x, final_scaler)

    # ------------------------------------------------------------------
    # STEPS 26N-26Q: final neural refit on reference dual-GPU backend.
    # ------------------------------------------------------------------
    visible_gpus = tf.config.list_physical_devices("GPU")
    reference_hardware = len(visible_gpus) == 2
    if not reference_hardware and not args.allow_nonreference_hardware:
        raise RuntimeError(
            "Reference protocol requires exactly 2 visible GPUs for final neural refit. "
            "Use --allow-nonreference-hardware only for a functional, non-reference-hardware run."
        )

    if reference_hardware:
        strategy = tf.distribute.MirroredStrategy()
        if strategy.num_replicas_in_sync != 2:
            raise RuntimeError("Expected two MirroredStrategy replicas.")
        backend_text = "2 x GPU / TensorFlow MirroredStrategy"
    else:
        strategy = tf.distribute.get_strategy()
        backend_text = "NON-REFERENCE HARDWARE / default TensorFlow strategy"

    neural_dirs = {
        "DNN": final_root / "04_dnn",
        "LSTM": final_root / "05_lstm",
        "CNN-LSTM": final_root / "06_cnn_lstm",
        "Attention-CNN-LSTM": final_root / "07_attention_cnn_lstm",
    }
    final_names = {
        "DNN": "dnn_final_full_training_dual_gpu.keras",
        "LSTM": "lstm_final_full_training_dual_gpu.keras",
        "CNN-LSTM": "cnn_lstm_final_full_training_dual_gpu.keras",
        "Attention-CNN-LSTM": "attention_cnn_lstm_final_full_training_dual_gpu.keras",
    }
    stems = {
        "DNN": "dnn", "LSTM": "lstm", "CNN-LSTM": "cnn_lstm",
        "Attention-CNN-LSTM": "attention_cnn_lstm",
    }

    final_neural_rows = []
    for model_name, epochs in EXPECTED_FINAL_EPOCHS.items():
        model_dir = neural_dirs[model_name]
        model_dir.mkdir(parents=True, exist_ok=True)
        tf.keras.backend.clear_session()
        set_seed(PRIMARY_SEED)
        with strategy.scope():
            model = build_model(model_name)
        if model.count_params() != EXPECTED_PARAMS[model_name]:
            raise RuntimeError(f"{model_name} final parameter count mismatch.")

        train_input = neural_shape(x_full_neural, model_name)
        start = time.perf_counter()
        history = model.fit(
            train_input,
            y_full_np,
            epochs=epochs,
            batch_size=GLOBAL_BATCH_SIZE,
            class_weight=class_weight,
            callbacks=[],
            validation_data=None,
            shuffle=True,
            verbose=0,
        )
        train_time = time.perf_counter() - start
        model_path = model_dir / final_names[model_name]
        model.save(model_path)
        pd.DataFrame(history.history).to_csv(
            model_dir / f"{stems[model_name]}_final_full_training_history.csv", index=False
        )

        config = {
            "model": model_name,
            "training_rows": EXPECTED_FULL_ROWS,
            "features": FEATURES,
            "input_dimension": 16,
            "preprocessing": "5-way proto one-hot + 11 continuous RobustScaler(25,75)",
            "scaler_fit_scope": "ALL official Training.csv rows only",
            "shared_scaler_sha256": sha256_file(final_scaler_path),
            "seed": PRIMARY_SEED,
            "epochs": epochs,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "num_replicas": int(strategy.num_replicas_in_sync),
            "backend": backend_text,
            "reference_hardware": bool(reference_hardware),
            "mixed_precision": False,
            "class_weight": {"Normal": NORMAL_WEIGHT, "Attack": ATTACK_WEIGHT},
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "loss": "BinaryCrossentropy",
            "early_stopping": False,
            "validation_monitoring": False,
            "decision_threshold": THRESHOLD,
            "testing_csv_accessed": False,
            "model_parameters": EXPECTED_PARAMS[model_name],
            "model_sha256": sha256_file(model_path),
        }
        save_json(model_dir / f"{stems[model_name]}_final_refit_configuration.json", config)
        result = {
            "Model": model_name,
            "Training_Rows": EXPECTED_FULL_ROWS,
            "Epochs": epochs,
            "Training_Time_sec": float(train_time),
            "Model_Parameters": EXPECTED_PARAMS[model_name],
            "Decision_Threshold": THRESHOLD,
            "Testing_csv_accessed": False,
            "Reference_Hardware": bool(reference_hardware),
            "Model_SHA256": sha256_file(model_path),
        }
        save_json(model_dir / f"{stems[model_name]}_final_refit_result.json", result)
        final_neural_rows.append(result)
        del model, history
        gc.collect()
        tf.keras.backend.clear_session()

    # ------------------------------------------------------------------
    # Frozen protocol manifest.
    # ------------------------------------------------------------------
    save_json(root / "common_protocol_frozen_refit_protocol.json", {
        "status": "FROZEN_REPRODUCTION_PROTOCOL",
        "split": {
            "train_rows": EXPECTED_TRAIN_ROWS,
            "validation_rows": EXPECTED_VAL_ROWS,
            "train_sha256": train_sha,
            "validation_sha256": val_sha,
        },
        "class_weight": {"Normal": NORMAL_WEIGHT, "Attack": ATTACK_WEIGHT},
        "decision_threshold": THRESHOLD,
        "deep_development_seeds": DEEP_SEEDS,
        "primary_deep_seed": PRIMARY_SEED,
        "frozen_final_epochs": EXPECTED_FINAL_EPOCHS,
        "final_neural_scaler_sha256": sha256_file(final_scaler_path),
        "final_neural_backend": backend_text,
        "reference_hardware": bool(reference_hardware),
        "testing_csv_accessed": False,
        "post_test_tuning_allowed": False,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "tensorflow": tf.__version__,
            "lightgbm": lgb.__version__,
            "xgboost": xgb.__version__,
            "determinism": determinism,
        },
    })
    pd.DataFrame(final_neural_rows).to_csv(root / "final_neural_refit_summary.csv", index=False)

    print("Common-protocol comparator reproduction complete.")
    print("STEP-26A split SHA256 verified:", train_sha, val_sha)
    print("Frozen primary seed-42 epochs:", neural_primary_epochs)
    print("Final neural backend:", backend_text)
    print("Testing.csv accessed: NO")


if __name__ == "__main__":
    main()
