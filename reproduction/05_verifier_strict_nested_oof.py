from __future__ import annotations
import argparse, gc, os, random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler

from common import clean_bot_iot_csv, FEATURES
from sra_ids.verifier import build_verifier, signed_log1p

LOG_FEATURES = [
    "sport","dport","stddev","N_IN_Conn_P_SrcIP","min","state_number",
    "mean","N_IN_Conn_P_DstIP","drate","srate","max","NN1_distance","NN3_distance"
]
DIRECT_FEATURES = ["primary_p_attack", "leaf_match_rate"]

def numeric_transform(frame):
    out = np.empty((len(frame), 15), dtype=np.float64)
    for j, c in enumerate(LOG_FEATURES):
        out[:, j] = signed_log1p(frame[c].to_numpy(dtype=np.float64))
    for j, c in enumerate(DIRECT_FEATURES):
        out[:, len(LOG_FEATURES)+j] = frame[c].to_numpy(dtype=np.float64)
    return out

def fit_scaler(frame):
    s = RobustScaler(quantile_range=(10.0, 90.0))
    s.fit(numeric_transform(frame))
    return s

def transform(frame, scaler):
    cont = scaler.transform(numeric_transform(frame)).astype(np.float32)
    proto = frame["proto"].to_numpy(dtype=np.int32)
    onehot = np.zeros((len(frame),5), dtype=np.float32)
    onehot[np.arange(len(frame)), proto] = 1.0
    return np.concatenate([onehot, cont], axis=1).astype(np.float32)

def balanced_weights(y):
    y = np.asarray(y, dtype=np.uint8)
    n, n0, n1 = len(y), int((y==0).sum()), int((y==1).sum())
    return {0: n/(2*n0), 1: n/(2*n1)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", required=True)
    ap.add_argument("--split-npz", required=True)
    ap.add_argument("--context-npz", required=True)
    ap.add_argument("--verifier-index-npz", required=True)
    ap.add_argument("--output-dir", default="reproduction_outputs/05_verifier_oof")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    os.environ["PYTHONHASHSEED"] = "42"
    random.seed(42); np.random.seed(42); tf.keras.utils.set_random_seed(42)
    try: tf.config.experimental.enable_op_determinism()
    except Exception: pass

    x, y = clean_bot_iot_csv(args.training_csv)
    split = np.load(args.split_npz)
    train_idx = split["train_idx"]
    x_train = x.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)

    ctx = np.load(args.context_npz)
    idx = np.load(args.verifier_index_npz)
    source = idx["source_train_index"].astype(np.int64)
    fold_v = idx["verifier_fold_id"].astype(np.int8)

    vf = x_train.iloc[source].reset_index(drop=True).copy()
    vf["primary_p_attack"] = ctx["primary_p_attack"][source]
    vf["leaf_match_rate"] = ctx["leaf_match_rate"][source]
    vf["NN1_distance"] = ctx["NN1_distance"][source]
    vf["NN3_distance"] = ctx["NN3_distance"][source]
    y_v = y_train.iloc[source].to_numpy(dtype=np.uint8)

    oof = np.full(len(vf), np.nan, dtype=np.float64)
    rows = []
    for outer in range(5):
        inner = (outer + 1) % 5
        outer_idx = np.flatnonzero(fold_v == outer)
        inner_idx = np.flatnonzero(fold_v == inner)
        selector_idx = np.flatnonzero((fold_v != outer) & (fold_v != inner))
        final_idx = np.flatnonzero(fold_v != outer)

        selector_scaler = fit_scaler(vf.iloc[selector_idx])
        xs = transform(vf.iloc[selector_idx], selector_scaler)
        xi = transform(vf.iloc[inner_idx], selector_scaler)
        ys, yi = y_v[selector_idx], y_v[inner_idx]

        tf.keras.backend.clear_session()
        seed = 42 + outer
        tf.keras.utils.set_random_seed(seed)
        selector_model = build_verifier(input_dim=20, learning_rate=1e-3)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=25,
                min_delta=1e-5, restore_best_weights=True, verbose=0
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", mode="min", factor=0.5,
                patience=10, min_lr=1e-5, verbose=0
            ),
        ]
        hist = selector_model.fit(
            xs, ys, validation_data=(xi, yi),
            epochs=300, batch_size=64, class_weight=balanced_weights(ys),
            callbacks=callbacks, verbose=0, shuffle=True
        )
        val_loss = np.asarray(hist.history["val_loss"], dtype=np.float64)
        selected_epoch = int(np.argmin(val_loss) + 1)
        best_val_loss = float(val_loss[selected_epoch-1])
        del selector_model, hist, xs, xi, selector_scaler
        gc.collect(); tf.keras.backend.clear_session()

        final_scaler = fit_scaler(vf.iloc[final_idx])
        xf = transform(vf.iloc[final_idx], final_scaler)
        xo = transform(vf.iloc[outer_idx], final_scaler)

        tf.keras.utils.set_random_seed(seed)
        model = build_verifier(input_dim=20, learning_rate=1e-3)
        model.fit(
            xf, y_v[final_idx], epochs=selected_epoch, batch_size=64,
            class_weight=balanced_weights(y_v[final_idx]), verbose=0, shuffle=True
        )
        oof[outer_idx] = model.predict(xo, batch_size=64, verbose=0).reshape(-1)
        rows.append({
            "outer_fold": outer, "inner_fold": inner,
            "selected_epoch": selected_epoch,
            "best_inner_val_loss": best_val_loss,
            "outer_rows": len(outer_idx),
        })
        del model, xf, xo, final_scaler
        gc.collect(); tf.keras.backend.clear_session()

    if np.isnan(oof).any():
        raise RuntimeError("Missing verifier OOF probabilities.")

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(out / "verifier_nested_oof_fold_results.csv", index=False)
    selected_epochs = fold_df["selected_epoch"].to_numpy(dtype=np.int32)
    np.savez_compressed(
        out / "verifier_strict_oof.npz",
        verifier_p_attack=oof,
        source_train_index=source,
        verifier_fold_id=fold_v,
        y_verifier=y_v,
        selected_epochs=selected_epochs,
    )
    median_epoch = int(np.median(selected_epochs))
    if median_epoch != 29:
        raise RuntimeError(f"Expected frozen median epoch 29, observed {median_epoch}")
    print("Strict verifier OOF complete. Selected epochs:", fold_df["selected_epoch"].tolist())
    print("Median frozen epoch:", median_epoch)

if __name__ == "__main__":
    main()
