from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler

from common import FEATURES, NUMERIC_FEATURES, clean_bot_iot_csv, save_json
from sra_ids.verifier import build_verifier, signed_log1p

PRIMARY_BASE_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=42,
    n_jobs=-1,
    deterministic=True,
    force_col_wise=True,
    verbosity=-1,
)
PRIMARY_NORMAL_WEIGHT = 400.0
PRIMARY_THRESHOLD = 0.50
EXPECTED_PRIMARY_TREES = 125
EXPECTED_VERIFIER_EPOCHS = 29
EXPECTED_ROUTER_THRESHOLD = 1.145083189010620
EXPECTED_VERIFIER_THRESHOLD = 0.40
EXPECTED_VERIFIER_PARAMS = 18_945

LOG_FEATURES = [
    "sport",
    "dport",
    "stddev",
    "N_IN_Conn_P_SrcIP",
    "min",
    "state_number",
    "mean",
    "N_IN_Conn_P_DstIP",
    "drate",
    "srate",
    "max",
    "NN1_distance",
    "NN3_distance",
]
DIRECT_FEATURES = ["primary_p_attack", "leaf_match_rate"]


def router_transform(frame, scaler):
    numeric = scaler.transform(frame[NUMERIC_FEATURES]).astype(np.float32)
    proto = frame["proto"].to_numpy(dtype=np.int32)
    if np.any((proto < 0) | (proto > 4)):
        raise RuntimeError("Unexpected protocol code outside 0..4.")
    onehot = np.zeros((len(frame), 5), dtype=np.float32)
    onehot[np.arange(len(frame)), proto] = 1.0
    return np.concatenate([numeric, onehot], axis=1)


def verifier_numeric(frame):
    out = np.empty((len(frame), 15), dtype=np.float64)
    for j, column in enumerate(LOG_FEATURES):
        out[:, j] = signed_log1p(frame[column].to_numpy(dtype=np.float64))
    for j, column in enumerate(DIRECT_FEATURES):
        out[:, len(LOG_FEATURES) + j] = frame[column].to_numpy(dtype=np.float64)
    return out


def verifier_transform(frame, scaler):
    continuous = scaler.transform(verifier_numeric(frame)).astype(np.float32)
    proto = frame["proto"].to_numpy(dtype=np.int32)
    onehot = np.zeros((len(frame), 5), dtype=np.float32)
    onehot[np.arange(len(frame)), proto] = 1.0
    return np.concatenate([onehot, continuous], axis=1).astype(np.float32)


def balanced_weights(y):
    y = np.asarray(y, dtype=np.uint8)
    n = len(y)
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    if n0 == 0 or n1 == 0:
        raise RuntimeError("Verifier training subset must contain both classes.")
    return {0: n / (2.0 * n0), 1: n / (2.0 * n1)}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", required=True)
    ap.add_argument("--split-npz", required=True)
    ap.add_argument("--primary-oof-npz", required=True)
    ap.add_argument("--context-npz", required=True)
    ap.add_argument("--verifier-index-npz", required=True)
    ap.add_argument("--verifier-oof-npz", required=True)
    ap.add_argument("--router-json", required=True)
    ap.add_argument("--verifier-threshold-json", required=True)
    ap.add_argument("--output-dir", default="reproduction_outputs/07_final_sra")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Freeze all final choices from their preceding strict-OOF artifacts.
    # This mirrors STEP 19: derive, then safety-check, rather than hard-code.
    # ------------------------------------------------------------------
    primary_oof = np.load(args.primary_oof_npz, allow_pickle=False)
    primary_iterations = primary_oof["selected_iterations"].astype(np.int32)
    final_primary_trees = int(np.median(primary_iterations))
    if final_primary_trees != EXPECTED_PRIMARY_TREES:
        raise RuntimeError(
            f"Unexpected median primary tree count: {final_primary_trees}"
        )

    verifier_oof = np.load(args.verifier_oof_npz, allow_pickle=False)
    if "selected_epochs" not in verifier_oof.files:
        raise RuntimeError(
            "verifier_strict_oof.npz lacks selected_epochs; rerun corrected STEP 05."
        )
    verifier_epochs = verifier_oof["selected_epochs"].astype(np.int32)
    final_verifier_epochs = int(np.median(verifier_epochs))
    if final_verifier_epochs != EXPECTED_VERIFIER_EPOCHS:
        raise RuntimeError(
            f"Unexpected median verifier epoch count: {final_verifier_epochs}"
        )

    router_cfg = load_json(args.router_json)
    router_probability_threshold = float(
        router_cfg.get("primary_probability_threshold", PRIMARY_THRESHOLD)
    )
    router_nn1_threshold = float(router_cfg["NN1_distance_threshold"])
    if not np.isclose(router_probability_threshold, PRIMARY_THRESHOLD):
        raise RuntimeError("Unexpected frozen primary routing threshold.")
    if not np.isclose(
        router_nn1_threshold, EXPECTED_ROUTER_THRESHOLD, atol=1e-12
    ):
        raise RuntimeError("Unexpected frozen NN1 routing threshold.")

    threshold_cfg = load_json(args.verifier_threshold_json)
    final_verifier_threshold = float(threshold_cfg["threshold"])
    if not np.isclose(final_verifier_threshold, EXPECTED_VERIFIER_THRESHOLD):
        raise RuntimeError("Unexpected frozen verifier threshold.")

    x, y = clean_bot_iot_csv(args.training_csv)
    split = np.load(args.split_npz, allow_pickle=False)
    train_idx = split["train_idx"].astype(np.int64)
    x_train = x.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_np = y_train.to_numpy(dtype=np.uint8)

    # ------------------------------------------------------------------
    # Final Primary LightGBM on the frozen development-training partition.
    # ------------------------------------------------------------------
    primary_params = dict(PRIMARY_BASE_PARAMS, n_estimators=final_primary_trees)
    primary_sample_weight = np.where(
        y_np == 0, PRIMARY_NORMAL_WEIGHT, 1.0
    ).astype(np.float32)
    primary = lgb.LGBMClassifier(**primary_params)
    primary.fit(
        x_train,
        y_train,
        sample_weight=primary_sample_weight,
        categorical_feature=["proto"],
    )
    if int(primary.booster_.num_trees()) != final_primary_trees:
        raise RuntimeError("Unexpected fitted final Primary tree count.")
    primary.booster_.save_model(str(out / "final_primary_lightgbm.txt"))

    # ------------------------------------------------------------------
    # Final Normal-reference NN router and Normal-leaf structures.
    # ------------------------------------------------------------------
    router_scaler = StandardScaler()
    router_scaler.fit(x_train[NUMERIC_FEATURES])
    normal_idx = np.flatnonzero(y_np == 0)
    if len(normal_idx) != 296:
        raise RuntimeError(f"Expected 296 Normal references, observed {len(normal_idx)}.")

    normal_reference_x = router_transform(x_train.iloc[normal_idx], router_scaler)
    nn_model = NearestNeighbors(
        n_neighbors=3,
        algorithm="ball_tree",
        metric="euclidean",
        n_jobs=-1,
    )
    nn_model.fit(normal_reference_x)

    joblib.dump(router_scaler, out / "final_router_standard_scaler.joblib")
    joblib.dump(nn_model, out / "final_normal_reference_nn.joblib")
    np.save(out / "final_normal_reference_X.npy", normal_reference_x)

    normal_leaf_matrix = np.asarray(
        primary.booster_.predict(x_train.iloc[normal_idx][FEATURES], pred_leaf=True)
    )
    if normal_leaf_matrix.ndim == 1:
        normal_leaf_matrix = normal_leaf_matrix[:, None]
    if normal_leaf_matrix.shape[1] != final_primary_trees:
        raise RuntimeError("Final Normal-leaf tree-count mismatch.")
    normal_leaf_sets = [
        np.unique(normal_leaf_matrix[:, tree_id])
        for tree_id in range(final_primary_trees)
    ]
    joblib.dump(normal_leaf_sets, out / "final_normal_leaf_sets.joblib")

    # ------------------------------------------------------------------
    # Final verifier development set from strict-OOF routing only.
    # ------------------------------------------------------------------
    context = np.load(args.context_npz, allow_pickle=False)
    verifier_index = np.load(args.verifier_index_npz, allow_pickle=False)
    source = verifier_index["source_train_index"].astype(np.int64)
    if len(source) != 1_139:
        raise RuntimeError(f"Expected 1,139 routed verifier rows, observed {len(source)}.")

    verifier_frame = x_train.iloc[source].reset_index(drop=True).copy()
    verifier_frame["primary_p_attack"] = context["primary_p_attack"][source]
    verifier_frame["leaf_match_rate"] = context["leaf_match_rate"][source]
    verifier_frame["NN1_distance"] = context["NN1_distance"][source]
    verifier_frame["NN3_distance"] = context["NN3_distance"][source]
    y_verifier = y_np[source]

    if int((y_verifier == 0).sum()) != 295 or int((y_verifier == 1).sum()) != 844:
        raise RuntimeError("Unexpected final verifier class counts.")

    verifier_scaler = RobustScaler(quantile_range=(10.0, 90.0))
    verifier_scaler.fit(verifier_numeric(verifier_frame))
    x_verifier = verifier_transform(verifier_frame, verifier_scaler)
    if x_verifier.shape != (1_139, 20):
        raise RuntimeError(f"Unexpected verifier matrix shape: {x_verifier.shape}")
    joblib.dump(verifier_scaler, out / "final_verifier_robust_scaler.joblib")

    # ------------------------------------------------------------------
    # Final residual-attention verifier: exactly median 29 epochs;
    # no early stopping and no validation/test adaptation.
    # ------------------------------------------------------------------
    os.environ["PYTHONHASHSEED"] = "42"
    random.seed(42)
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    verifier_model = build_verifier(input_dim=20, learning_rate=1e-3)
    if int(verifier_model.count_params()) != EXPECTED_VERIFIER_PARAMS:
        raise RuntimeError(
            f"Unexpected verifier parameter count: {verifier_model.count_params()}"
        )

    class_weight = balanced_weights(y_verifier)
    history = verifier_model.fit(
        x_verifier,
        y_verifier,
        epochs=final_verifier_epochs,
        batch_size=64,
        class_weight=class_weight,
        verbose=0,
        shuffle=True,
    )
    verifier_model.save(out / "final_residual_attention.keras")
    verifier_model.save_weights(out / "final_residual_attention.weights.h5")

    save_json(
        out / "frozen_hybrid_configuration.json",
        {
            "primary_selected_iterations": primary_iterations.tolist(),
            "primary_tree_selection_rule": "median of strict STEP-02 outer-fold selections",
            "primary_trees": final_primary_trees,
            "primary_threshold": PRIMARY_THRESHOLD,
            "primary_normal_weight": PRIMARY_NORMAL_WEIGHT,
            "router_probability_rule": f"p_attack < {router_probability_threshold:.2f}",
            "router_NN1_rule": (
                f"NN1_distance <= {router_nn1_threshold:.15f}"
            ),
            "normal_reference_count": int(len(normal_idx)),
            "verifier_selected_epochs": verifier_epochs.tolist(),
            "verifier_epoch_selection_rule": "median of strict STEP-05 outer-fold selections",
            "verifier_epochs": final_verifier_epochs,
            "verifier_input_dim": 20,
            "verifier_parameters": EXPECTED_VERIFIER_PARAMS,
            "verifier_class_weight": {
                "Normal": float(class_weight[0]),
                "Attack": float(class_weight[1]),
            },
            "verifier_threshold": final_verifier_threshold,
            "final_verifier_early_stopping": False,
            "external_validation_used": False,
            "testing_csv_accessed": False,
            "test_based_tuning": False,
            "training_history_epochs": int(len(history.history.get("loss", []))),
        },
    )

    print("Final frozen SRA-IDS artifacts written to", out)
    print("Primary trees derived from strict OOF:", primary_iterations.tolist(), "->", final_primary_trees)
    print("Verifier epochs derived from strict OOF:", verifier_epochs.tolist(), "->", final_verifier_epochs)


if __name__ == "__main__":
    main()
