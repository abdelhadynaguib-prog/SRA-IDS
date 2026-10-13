from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

FEATURES = [
    "proto", "sport", "dport", "stddev", "N_IN_Conn_P_SrcIP", "min",
    "state_number", "mean", "N_IN_Conn_P_DstIP", "drate", "srate", "max",
]
NUMERIC_FEATURES = FEATURES[1:]
PROTO_MAP = {"arp": 0, "icmp": 1, "ipv6-icmp": 2, "tcp": 3, "udp": 4}
TARGET = "attack"
RANDOM_STATE = 42


def parse_port_column(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    low = text.str.lower()
    hex_mask = low.str.match(r"(?i)^0x[0-9a-f]+$", na=False)
    out = pd.to_numeric(text.where(~hex_mask), errors="coerce")
    if hex_mask.any():
        vals = [int(v, 16) for v in low[hex_mask]]
        out.loc[hex_mask] = vals
    if out.isna().any():
        bad = text[out.isna()].head(10).tolist()
        raise ValueError(f"Unparseable port values: {bad}")
    values = out.to_numpy(dtype=np.float64)
    if not np.all(np.isclose(values, np.round(values))):
        raise ValueError("Non-integer port values detected after cleaning.")
    return pd.Series(np.round(values).astype(np.int32), index=series.index)


def clean_bot_iot_csv(path: str | Path):
    usecols = FEATURES + [TARGET]
    raw = pd.read_csv(path, usecols=usecols, low_memory=False)
    x = raw[FEATURES].copy()

    proto = x["proto"].astype(str).str.strip().str.lower()
    unknown = sorted(set(proto.unique()) - set(PROTO_MAP))
    if unknown:
        raise ValueError(f"Unknown protocol values: {unknown}")
    x["proto"] = proto.map(PROTO_MAP).astype(np.int8)

    x["sport"] = parse_port_column(x["sport"])
    x["dport"] = parse_port_column(x["dport"])
    for col in NUMERIC_FEATURES[2:]:
        x[col] = pd.to_numeric(x[col], errors="raise")

    if not np.isfinite(x.to_numpy(dtype=np.float64)).all():
        raise ValueError("Non-finite feature values detected.")

    # Original STEP 2 freezes one deterministic downstream representation:
    # all 12 predictors are stored as float32 before any duplicate grouping,
    # OOF construction, overlap audit, or model training.
    x = x[FEATURES].astype(np.float32)

    y = pd.to_numeric(raw[TARGET], errors="raise").astype(np.uint8)
    if sorted(y.unique().tolist()) != [0, 1]:
        raise ValueError("Expected binary attack labels 0/1.")
    return x, y


def build_exact_group_ids(x: pd.DataFrame) -> np.ndarray:
    """Exact-vector group IDs: duplicates share a group; singletons remain unique."""
    n = len(x)
    duplicate_mask = x.duplicated(subset=FEATURES, keep=False).to_numpy()
    group_id = np.arange(n, dtype=np.int64)

    if duplicate_mask.any():
        dup_df = x.loc[duplicate_mask, FEATURES]
        dup_codes = (
            dup_df.groupby(FEATURES, sort=False, observed=True, dropna=False)
            .ngroup()
            .to_numpy(dtype=np.int64)
        )
        group_id[duplicate_mask] = n + dup_codes
    return group_id


def create_development_split(x: pd.DataFrame, y: pd.Series, seed: int = 42):
    """Rebuild the leakage-safe 80/20 development split used by the study."""
    n = len(x)
    y_np = y.to_numpy(dtype=np.uint8)
    group_id = build_exact_group_ids(x)

    # Label consistency for exact duplicate vectors.
    dup_mask = x.duplicated(subset=FEATURES, keep=False).to_numpy()
    if dup_mask.any():
        check = pd.DataFrame({"group": group_id[dup_mask], "label": y_np[dup_mask]})
        if int(check.groupby("group")["label"].nunique().max()) != 1:
            raise RuntimeError("Contradictory labels inside an exact duplicate group.")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    dummy = np.zeros(n, dtype=np.uint8)
    fold_id = np.full(n, -1, dtype=np.int8)
    stats = []

    for fold, (_, val_idx) in enumerate(sgkf.split(dummy, y_np, groups=group_id)):
        fold_id[val_idx] = fold
        fy = y_np[val_idx]
        stats.append({
            "fold": fold,
            "rows": len(val_idx),
            "normal": int((fy == 0).sum()),
            "attack": int((fy == 1).sum()),
        })

    target_normal = int(round(int((y_np == 0).sum()) / 5))
    target_rows = n / 5.0
    selected = min(
        stats,
        key=lambda z: (
            abs(z["normal"] - target_normal),
            abs(z["rows"] - target_rows),
            z["fold"],
        ),
    )
    val_fold = int(selected["fold"])
    val_idx = np.flatnonzero(fold_id == val_fold)
    train_idx = np.flatnonzero(fold_id != val_fold)

    # Exact-group leakage check.
    train_groups = np.unique(group_id[train_idx])
    val_groups = np.unique(group_id[val_idx])
    if np.intersect1d(train_groups, val_groups).size:
        raise RuntimeError("Exact-vector group leakage detected.")

    return train_idx, val_idx, val_fold, fold_id, stats


def _class_fold_targets(total: int, n_splits: int = 5) -> np.ndarray:
    base = total // n_splits
    rem = total % n_splits
    out = np.full(n_splits, base, dtype=np.int64)
    out[:rem] += 1
    return out


def create_strict_oof_fold_ids(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 42,
    n_splits: int = 5,
) -> np.ndarray:
    """Exact class-balanced, duplicate-disjoint 5-fold assignment used for strict OOF."""
    y_np = y_train.to_numpy(dtype=np.uint8)
    n = len(x_train)

    duplicate_mask = x_train.duplicated(subset=FEATURES, keep=False).to_numpy()
    dup_positions = np.flatnonzero(duplicate_mask)

    if len(dup_positions):
        dup_features = x_train.iloc[dup_positions][FEATURES]
        dup_group_code = (
            dup_features.groupby(FEATURES, sort=False, observed=True, dropna=False)
            .ngroup().to_numpy(dtype=np.int32)
        )
        dup_meta = pd.DataFrame({
            "row": dup_positions,
            "group": dup_group_code,
            "label": y_np[dup_positions],
        })
        group_table = (
            dup_meta.groupby("group", sort=False)
            .agg(label=("label", "first"), size=("label", "size"), n_labels=("label", "nunique"))
            .reset_index()
        )
        if int((group_table["n_labels"] > 1).sum()) != 0:
            raise RuntimeError("Contradictory labels in duplicate groups.")
        group_rows = {
            int(group): rows["row"].to_numpy(dtype=np.int64)
            for group, rows in dup_meta.groupby("group", sort=False)
        }
    else:
        group_table = pd.DataFrame(columns=["group", "label", "size", "n_labels"])
        group_rows = {}

    normal_targets = _class_fold_targets(int((y_np == 0).sum()), n_splits)
    attack_targets = _class_fold_targets(int((y_np == 1).sum()), n_splits)

    rng = np.random.default_rng(seed)
    fold_id = np.full(n, -1, dtype=np.int8)

    def assign_class(class_label: int, targets: np.ndarray):
        current = np.zeros(n_splits, dtype=np.int64)
        class_groups = group_table[group_table["label"] == class_label].copy()

        if len(class_groups):
            class_groups["_random"] = rng.random(len(class_groups))
            class_groups = class_groups.sort_values(
                ["size", "_random"], ascending=[False, True]
            )
            for row in class_groups.itertuples():
                size = int(row.size)
                remaining = targets - current
                feasible = np.flatnonzero(remaining >= size)
                if len(feasible) == 0:
                    raise RuntimeError(
                        f"Could not place duplicate group {row.group} for class {class_label}."
                    )
                best_fold = int(feasible[np.argmax(remaining[feasible])])
                rows = group_rows[int(row.group)]
                fold_id[rows] = best_fold
                current[best_fold] += size

        singleton_rows = np.flatnonzero((~duplicate_mask) & (y_np == class_label))
        singleton_rows = rng.permutation(singleton_rows)
        needed = targets - current
        if np.any(needed < 0) or int(needed.sum()) != len(singleton_rows):
            raise RuntimeError(f"Singleton quota mismatch for class {class_label}.")
        start = 0
        for fold in range(n_splits):
            end = start + int(needed[fold])
            fold_id[singleton_rows[start:end]] = fold
            start = end

    assign_class(0, normal_targets)
    assign_class(1, attack_targets)

    if np.any(fold_id < 0):
        raise RuntimeError("Some training rows were not assigned to an OOF fold.")

    for f in range(n_splits):
        mask = fold_id == f
        if int(((y_np == 0) & mask).sum()) != int(normal_targets[f]):
            raise RuntimeError("Normal OOF quota mismatch.")
        if int(((y_np == 1) & mask).sum()) != int(attack_targets[f]):
            raise RuntimeError("Attack OOF quota mismatch.")

    return fold_id


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
