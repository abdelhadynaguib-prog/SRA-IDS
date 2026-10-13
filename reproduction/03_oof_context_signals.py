from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from common import FEATURES, NUMERIC_FEATURES, clean_bot_iot_csv

COMMON_PARAMS = dict(
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
NORMAL_WEIGHT = 400.0
CHUNK_SIZE = 50_000
EXPECTED_ITERATIONS = [125, 209, 113, 249, 114]


def router_transform(frame: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    numeric = scaler.transform(frame[NUMERIC_FEATURES]).astype(np.float32)
    proto = frame["proto"].to_numpy(dtype=np.int32)
    if np.any((proto < 0) | (proto > 4)):
        raise RuntimeError("Unexpected protocol code outside 0..4.")
    onehot = np.zeros((len(frame), 5), dtype=np.float32)
    onehot[np.arange(len(frame)), proto] = 1.0
    return np.concatenate([numeric, onehot], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", required=True)
    ap.add_argument("--split-npz", required=True)
    ap.add_argument("--primary-oof-npz", required=True)
    ap.add_argument("--output-dir", default="reproduction_outputs/03_context")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    x, y = clean_bot_iot_csv(args.training_csv)
    split = np.load(args.split_npz, allow_pickle=False)
    train_idx = split["train_idx"].astype(np.int64)
    fold_id = split["strict_oof_fold_id"].astype(np.int8)

    primary = np.load(args.primary_oof_npz, allow_pickle=False)
    selected_iterations = primary["selected_iterations"].astype(np.int32)
    p_oof = primary["p_attack"].astype(np.float64)

    if selected_iterations.tolist() != EXPECTED_ITERATIONS:
        raise RuntimeError(
            f"Unexpected strict-OOF tree counts: {selected_iterations.tolist()}"
        )

    x_train = x.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_np = y_train.to_numpy(dtype=np.uint8)
    n = len(x_train)

    if len(fold_id) != n or len(p_oof) != n:
        raise RuntimeError("Split/OOF alignment mismatch.")

    leaf_rate = np.full(n, np.nan, dtype=np.float32)
    leaf_count = np.full(n, -1, dtype=np.int16)
    tree_count = np.full(n, -1, dtype=np.int16)
    nn1 = np.full(n, np.nan, dtype=np.float32)
    nn3 = np.full(n, np.nan, dtype=np.float32)
    fold_rows: list[dict] = []

    for outer in range(5):
        outer_idx = np.flatnonzero(fold_id == outer)
        reference_pool_idx = np.flatnonzero(fold_id != outer)
        normal_reference_idx = reference_pool_idx[
            y_np[reference_pool_idx] == 0
        ]
        if len(normal_reference_idx) < 3:
            raise RuntimeError("Fewer than three Normal references in an outer fold.")

        selected_iteration = int(selected_iterations[outer])
        sample_weight = np.where(
            y_np[reference_pool_idx] == 0, NORMAL_WEIGHT, 1.0
        ).astype(np.float32)

        model = lgb.LGBMClassifier(
            **COMMON_PARAMS,
            n_estimators=selected_iteration,
        )
        model.fit(
            x_train.iloc[reference_pool_idx],
            y_train.iloc[reference_pool_idx],
            sample_weight=sample_weight,
            categorical_feature=["proto"],
        )
        booster = model.booster_

        # STEP 13: Normal-occupied leaves from non-outer Normal references only.
        normal_reference_leaf = np.asarray(
            booster.predict(
                x_train.iloc[normal_reference_idx][FEATURES], pred_leaf=True
            )
        )
        if normal_reference_leaf.ndim == 1:
            normal_reference_leaf = normal_reference_leaf[:, None]
        n_trees = int(normal_reference_leaf.shape[1])
        if n_trees != selected_iteration:
            raise RuntimeError("Leaf tree-count mismatch.")

        normal_leaf_sets = [
            np.unique(normal_reference_leaf[:, tree_idx])
            for tree_idx in range(n_trees)
        ]
        outer_match_count = np.zeros(len(outer_idx), dtype=np.int16)

        leaf_start = time.perf_counter()
        for start in range(0, len(outer_idx), args.chunk_size):
            end = min(start + args.chunk_size, len(outer_idx))
            chunk_global_idx = outer_idx[start:end]
            leaf_chunk = np.asarray(
                booster.predict(
                    x_train.iloc[chunk_global_idx][FEATURES], pred_leaf=True
                )
            )
            if leaf_chunk.ndim == 1:
                leaf_chunk = leaf_chunk[:, None]
            if leaf_chunk.shape[1] != n_trees:
                raise RuntimeError("Chunk leaf tree-count mismatch.")

            chunk_match_count = np.zeros(len(chunk_global_idx), dtype=np.int16)
            for tree_idx in range(n_trees):
                chunk_match_count += np.isin(
                    leaf_chunk[:, tree_idx],
                    normal_leaf_sets[tree_idx],
                    assume_unique=False,
                ).astype(np.int16)
            outer_match_count[start:end] = chunk_match_count
            del leaf_chunk, chunk_match_count
            gc.collect()

        outer_match_rate = (
            outer_match_count.astype(np.float32) / float(n_trees)
        )
        leaf_count[outer_idx] = outer_match_count
        tree_count[outer_idx] = n_trees
        leaf_rate[outer_idx] = outer_match_rate
        leaf_elapsed = time.perf_counter() - leaf_start

        # STEP 14: fold-specific StandardScaler and Normal-reference 3-NN.
        scaler = StandardScaler()
        scaler.fit(x_train.iloc[reference_pool_idx][NUMERIC_FEATURES])
        normal_reference_x = router_transform(
            x_train.iloc[normal_reference_idx], scaler
        )
        nn_model = NearestNeighbors(
            n_neighbors=3,
            algorithm="ball_tree",
            metric="euclidean",
            n_jobs=-1,
        )
        nn_model.fit(normal_reference_x)

        fold_nn1 = np.empty(len(outer_idx), dtype=np.float32)
        fold_nn3 = np.empty(len(outer_idx), dtype=np.float32)
        nn_start = time.perf_counter()
        for start in range(0, len(outer_idx), args.chunk_size):
            end = min(start + args.chunk_size, len(outer_idx))
            query_idx = outer_idx[start:end]
            query_x = router_transform(x_train.iloc[query_idx], scaler)
            distances, _ = nn_model.kneighbors(
                query_x, n_neighbors=3, return_distance=True
            )
            fold_nn1[start:end] = distances[:, 0].astype(np.float32)
            fold_nn3[start:end] = distances.mean(axis=1).astype(np.float32)
            del query_x, distances
            gc.collect()

        nn1[outer_idx] = fold_nn1
        nn3[outer_idx] = fold_nn3
        nn_elapsed = time.perf_counter() - nn_start

        outer_y = y_np[outer_idx]
        fold_rows.append(
            {
                "outer_fold": outer,
                "trees": n_trees,
                "reference_pool_rows": int(len(reference_pool_idx)),
                "training_normal_refs": int(len(normal_reference_idx)),
                "outer_rows": int(len(outer_idx)),
                "outer_normal": int((outer_y == 0).sum()),
                "outer_attack": int((outer_y == 1).sum()),
                "normal_leaf_match_mean": float(
                    outer_match_rate[outer_y == 0].mean()
                ),
                "attack_leaf_match_mean": float(
                    outer_match_rate[outer_y == 1].mean()
                ),
                "normal_nn1_median": float(np.median(fold_nn1[outer_y == 0])),
                "attack_nn1_median": float(np.median(fold_nn1[outer_y == 1])),
                "normal_nn3_median": float(np.median(fold_nn3[outer_y == 0])),
                "attack_nn3_median": float(np.median(fold_nn3[outer_y == 1])),
                "leaf_compute_time_sec": float(leaf_elapsed),
                "nn_compute_time_sec": float(nn_elapsed),
            }
        )

        del (
            model,
            booster,
            normal_reference_leaf,
            normal_leaf_sets,
            outer_match_count,
            outer_match_rate,
            scaler,
            normal_reference_x,
            nn_model,
            fold_nn1,
            fold_nn3,
            sample_weight,
        )
        gc.collect()

    for name, values in {
        "leaf_match_rate": leaf_rate,
        "NN1_distance": nn1,
        "NN3_distance": nn3,
    }.items():
        if np.isnan(values).any() or not np.isfinite(values).all():
            raise RuntimeError(f"Missing/non-finite values in {name}.")

    pd.DataFrame(fold_rows).to_csv(
        out / "strict_oof_context_fold_results.csv", index=False
    )
    np.savez_compressed(
        out / "strict_oof_context_signals.npz",
        primary_p_attack=p_oof,
        leaf_match_rate=leaf_rate,
        leaf_match_count=leaf_count,
        tree_count=tree_count,
        NN1_distance=nn1,
        NN3_distance=nn3,
        strict_oof_fold_id=fold_id,
    )
    print("Strict OOF context signals reproduced with chunked STEP 13/14 processing.")


if __name__ == "__main__":
    main()
