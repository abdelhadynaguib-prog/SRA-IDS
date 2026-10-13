from __future__ import annotations
import argparse, gc
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, matthews_corrcoef, balanced_accuracy_score

from common import clean_bot_iot_csv, FEATURES

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
MAX_ESTIMATORS = 500
EARLY_STOPPING_ROUNDS = 50
THRESHOLD = 0.50
EXPECTED_ITERATIONS = [125, 209, 113, 249, 114]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", required=True)
    ap.add_argument("--split-npz", required=True)
    ap.add_argument("--output-dir", default="reproduction_outputs/02_primary_oof")
    ap.add_argument("--no-expected-assert", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    x, y = clean_bot_iot_csv(args.training_csv)
    split = np.load(args.split_npz)
    train_idx = split["train_idx"]
    fold_id = split["strict_oof_fold_id"]
    x_train = x.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_np = y_train.to_numpy(dtype=np.uint8)

    oof = np.full(len(x_train), np.nan, dtype=np.float64)
    rows = []
    for outer in range(5):
        inner = (outer + 1) % 5
        outer_idx = np.flatnonzero(fold_id == outer)
        inner_idx = np.flatnonzero(fold_id == inner)
        selector_idx = np.flatnonzero((fold_id != outer) & (fold_id != inner))
        final_idx = np.flatnonzero(fold_id != outer)

        selector_w = np.where(y_np[selector_idx] == 0, NORMAL_WEIGHT, 1.0).astype(np.float32)
        inner_w = np.where(y_np[inner_idx] == 0, NORMAL_WEIGHT, 1.0).astype(np.float32)

        selector = lgb.LGBMClassifier(**COMMON_PARAMS, n_estimators=MAX_ESTIMATORS)
        selector.fit(
            x_train.iloc[selector_idx], y_train.iloc[selector_idx],
            sample_weight=selector_w,
            eval_set=[(x_train.iloc[inner_idx], y_train.iloc[inner_idx])],
            eval_metric="binary_logloss",
            eval_sample_weight=[inner_w],
            categorical_feature=["proto"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        selected_iteration = int(selector.best_iteration_)
        del selector; gc.collect()

        final_w = np.where(y_np[final_idx] == 0, NORMAL_WEIGHT, 1.0).astype(np.float32)
        model = lgb.LGBMClassifier(**COMMON_PARAMS, n_estimators=selected_iteration)
        model.fit(
            x_train.iloc[final_idx], y_train.iloc[final_idx],
            sample_weight=final_w, categorical_feature=["proto"]
        )
        p = model.predict_proba(x_train.iloc[outer_idx])[:, 1]
        oof[outer_idx] = p
        pred = (p >= THRESHOLD).astype(np.uint8)
        tn, fp, fn, tp = confusion_matrix(y_np[outer_idx], pred, labels=[0, 1]).ravel()
        rows.append({
            "outer_fold": outer, "inner_val_fold": inner,
            "selected_iteration": selected_iteration,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "MCC": float(matthews_corrcoef(y_np[outer_idx], pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_np[outer_idx], pred)),
        })
        del model; gc.collect()

    if np.isnan(oof).any():
        raise RuntimeError("Missing OOF probabilities.")

    fold_df = pd.DataFrame(rows)
    iterations = fold_df["selected_iteration"].astype(int).tolist()
    if not args.no_expected_assert and iterations != EXPECTED_ITERATIONS:
        raise RuntimeError(f"Selected iterations differ: {iterations} != {EXPECTED_ITERATIONS}")

    fold_df.to_csv(out / "primary_nested_oof_fold_results.csv", index=False)
    np.savez_compressed(
        out / "primary_strict_oof.npz",
        p_attack=oof,
        strict_oof_fold_id=fold_id,
        selected_iterations=np.asarray(iterations, dtype=np.int32),
    )

    pred = (oof >= THRESHOLD).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_np, pred, labels=[0,1]).ravel()
    print("Strict primary OOF:", dict(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp)))
    print("Selected iterations:", iterations)

if __name__ == "__main__":
    main()
