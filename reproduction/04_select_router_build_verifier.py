from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import FEATURES, clean_bot_iot_csv, save_json

PRIMARY_GATE = 0.50
TARGET_CAPTURE = 0.995
EXPECTED_THRESHOLD = 1.145083189010620
EXPECTED_ROUTED = 1_139
EXPECTED_ROUTED_NORMAL = 295
EXPECTED_ROUTED_ATTACK = 844
EXPECTED_REPEATED_ROWS = 12

META_FEATURES = [
    "primary_p_attack",
    "leaf_match_rate",
    "NN1_distance",
    "NN3_distance",
]
VERIFIER_FEATURES = FEATURES + META_FEATURES


def evaluate_route(
    base_route: np.ndarray,
    rescue_gate: np.ndarray,
    y: np.ndarray,
    family: str,
    threshold: float,
) -> dict:
    routed = base_route | rescue_gate
    normal = y == 0
    attack = y == 1
    missed_normal = normal & (~base_route)
    return {
        "family": family,
        "threshold": float(threshold),
        "routed_rows": int(routed.sum()),
        "added_rows_vs_base": int(routed.sum() - base_route.sum()),
        "traffic_pct": float(100.0 * routed.mean()),
        "routed_normal": int((routed & normal).sum()),
        "normal_capture_pct": float(
            100.0 * (routed & normal).sum() / normal.sum()
        ),
        "normal_bypassing": int((normal & ~routed).sum()),
        "rescued_high_normal": int((routed & missed_normal).sum()),
        "missed_high_normal": int((missed_normal & ~routed).sum()),
        "routed_attack": int((routed & attack).sum()),
        "attack_routing_pct": float(
            100.0 * (routed & attack).sum() / attack.sum()
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", required=True)
    ap.add_argument("--split-npz", required=True)
    ap.add_argument("--context-npz", required=True)
    ap.add_argument("--output-dir", default="reproduction_outputs/04_router")
    ap.add_argument("--no-expected-assert", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    x, y = clean_bot_iot_csv(args.training_csv)
    split = np.load(args.split_npz, allow_pickle=False)
    train_idx = split["train_idx"].astype(np.int64)
    x_train = x.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True).to_numpy(dtype=np.uint8)

    d = np.load(args.context_npz, allow_pickle=False)
    p = d["primary_p_attack"].astype(np.float64)
    leaf = d["leaf_match_rate"].astype(np.float64)
    nn1 = d["NN1_distance"].astype(np.float64)
    nn3 = d["NN3_distance"].astype(np.float64)
    fold_id = d["strict_oof_fold_id"].astype(np.int8)

    n = len(x_train)
    if not all(len(a) == n for a in [y_train, p, leaf, nn1, nn3, fold_id]):
        raise RuntimeError("Training/context alignment mismatch.")

    normal_mask = y_train == 0
    attack_mask = y_train == 1
    base_route = p < PRIMARY_GATE
    missed_normal_idx = np.flatnonzero(normal_mask & (~base_route))

    rows: list[dict] = []
    rows.append(
        {
            "family": "BASE_ONLY",
            "threshold": np.nan,
            "routed_rows": int(base_route.sum()),
            "added_rows_vs_base": 0,
            "traffic_pct": float(100.0 * base_route.mean()),
            "routed_normal": int((base_route & normal_mask).sum()),
            "normal_capture_pct": float(
                100.0 * (base_route & normal_mask).sum() / normal_mask.sum()
            ),
            "normal_bypassing": int((normal_mask & ~base_route).sum()),
            "rescued_high_normal": 0,
            "missed_high_normal": int(len(missed_normal_idx)),
            "routed_attack": int((base_route & attack_mask).sum()),
            "attack_routing_pct": float(
                100.0 * (base_route & attack_mask).sum() / attack_mask.sum()
            ),
        }
    )

    # Original STEP 15 evaluates exact rescue breakpoints observed among
    # Normal samples missed by the probability-only route.
    for threshold in np.unique(leaf[missed_normal_idx]):
        rows.append(
            evaluate_route(
                base_route, leaf >= threshold, y_train, "LEAF", float(threshold)
            )
        )
    for threshold in np.unique(nn1[missed_normal_idx]):
        rows.append(
            evaluate_route(
                base_route, nn1 <= threshold, y_train, "NN1", float(threshold)
            )
        )
    for threshold in np.unique(nn3[missed_normal_idx]):
        rows.append(
            evaluate_route(
                base_route, nn3 <= threshold, y_train, "NN3", float(threshold)
            )
        )

    rescue_table = (
        pd.DataFrame(rows)
        .sort_values(
            ["routed_normal", "routed_rows", "family"],
            ascending=[True, True, True],
        )
        .reset_index(drop=True)
    )
    rescue_table.to_csv(out / "router_exact_rescue_cost_table.csv", index=False)

    target_rows = []
    for target in [0.990, 0.995, 1.000]:
        required_normal = int(
            math.ceil(target * int(normal_mask.sum()) - 1e-12)
        )
        candidates = (
            rescue_table[rescue_table["routed_normal"] >= required_normal]
            .sort_values(["routed_rows", "family"], ascending=[True, True])
            .reset_index(drop=True)
        )
        if candidates.empty:
            raise RuntimeError(f"No routing rule reaches target {target}.")
        row = candidates.iloc[0].to_dict()
        row["coverage_target"] = target
        row["required_normal"] = required_normal
        target_rows.append(row)

    target_table = pd.DataFrame(target_rows)
    target_table.to_csv(out / "router_target_coverage_selection.csv", index=False)

    best = target_table[np.isclose(target_table["coverage_target"], TARGET_CAPTURE)].iloc[0]
    if str(best["family"]) != "NN1":
        raise RuntimeError(f"Frozen 99.5% route unexpectedly selected {best['family']}.")

    nn1_threshold = float(best["threshold"])
    nn1_gate = nn1 <= nn1_threshold
    routed = base_route | nn1_gate
    source_idx = np.flatnonzero(routed)

    if not args.no_expected_assert:
        if not np.isclose(nn1_threshold, EXPECTED_THRESHOLD, atol=1e-12):
            raise RuntimeError(
                f"NN1 threshold {nn1_threshold:.15f} != {EXPECTED_THRESHOLD:.15f}"
            )
        if int(routed.sum()) != EXPECTED_ROUTED:
            raise RuntimeError("Unexpected routed-row count.")
        if int((routed & normal_mask).sum()) != EXPECTED_ROUTED_NORMAL:
            raise RuntimeError("Unexpected routed Normal count.")
        if int((routed & attack_mask).sum()) != EXPECTED_ROUTED_ATTACK:
            raise RuntimeError("Unexpected routed Attack count.")

    route_reason = np.full(n, "NOT_ROUTED", dtype=object)
    route_reason[base_route & nn1_gate] = "BASE_AND_NN1"
    route_reason[base_route & ~nn1_gate] = "BASE_ONLY"
    route_reason[~base_route & nn1_gate] = "NN1_RESCUE"

    verifier_df = x_train.iloc[source_idx][FEATURES].reset_index(drop=True).copy()
    verifier_df["primary_p_attack"] = p[source_idx]
    verifier_df["leaf_match_rate"] = leaf[source_idx]
    verifier_df["NN1_distance"] = nn1[source_idx]
    verifier_df["NN3_distance"] = nn3[source_idx]
    verifier_df["label"] = y_train[source_idx]
    verifier_df["fold"] = fold_id[source_idx]
    verifier_df["source_train_index"] = source_idx
    verifier_df["route_reason"] = route_reason[source_idx]

    if verifier_df[VERIFIER_FEATURES].isna().any().any():
        raise RuntimeError("Missing values in verifier development features.")
    if not np.isfinite(
        verifier_df[VERIFIER_FEATURES].to_numpy(dtype=np.float64)
    ).all():
        raise RuntimeError("Non-finite values in verifier development features.")

    duplicate_mask = verifier_df.duplicated(
        subset=VERIFIER_FEATURES, keep=False
    )
    repeated_rows = int(duplicate_mask.sum())
    crossing_groups = 0
    contradictory_groups = 0
    if repeated_rows:
        dup = verifier_df.loc[
            duplicate_mask, VERIFIER_FEATURES + ["label", "fold"]
        ].copy()
        dup["duplicate_group"] = (
            dup.groupby(
                VERIFIER_FEATURES,
                sort=False,
                observed=True,
                dropna=False,
            )
            .ngroup()
            .to_numpy(dtype=np.int64)
        )
        crossing_groups = int(
            (dup.groupby("duplicate_group")["fold"].nunique() > 1).sum()
        )
        contradictory_groups = int(
            (dup.groupby("duplicate_group")["label"].nunique() > 1).sum()
        )

    if crossing_groups != 0 or contradictory_groups != 0:
        raise RuntimeError("Verifier duplicate isolation/label consistency failed.")
    if not args.no_expected_assert and repeated_rows != EXPECTED_REPEATED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_REPEATED_ROWS} repeated verifier rows, observed {repeated_rows}."
        )

    reason_table = pd.crosstab(
        verifier_df["route_reason"],
        verifier_df["label"].map({0: "Normal", 1: "Attack"}),
        margins=True,
        margins_name="Total",
    )
    reason_table.to_csv(out / "verifier_route_reason_distribution.csv")

    fold_distribution = (
        verifier_df.groupby("fold")["label"]
        .agg(
            rows="size",
            normal=lambda s: int((s == 0).sum()),
            attack=lambda s: int((s == 1).sum()),
        )
        .reset_index()
    )
    fold_distribution.to_csv(out / "verifier_fold_distribution.csv", index=False)
    verifier_df.to_csv(out / "verifier_development_dataset.csv", index=False)

    missed_normal = pd.DataFrame(
        {
            "train_index": np.flatnonzero(normal_mask & ~routed),
            "p_attack": p[normal_mask & ~routed],
            "leaf_match_rate": leaf[normal_mask & ~routed],
            "NN1_distance": nn1[normal_mask & ~routed],
            "NN3_distance": nn3[normal_mask & ~routed],
            "fold": fold_id[normal_mask & ~routed],
        }
    )
    missed_normal.to_csv(out / "normal_not_routed.csv", index=False)

    save_json(
        out / "frozen_router_selection.json",
        {
            "selection_criterion": "minimum workload at >=99.5% Normal capture",
            "primary_probability_rule": "p_attack < 0.50",
            "primary_probability_threshold": PRIMARY_GATE,
            "rescue_family": "NN1",
            "NN1_distance_threshold": nn1_threshold,
            "final_rule": (
                f"(p_attack < 0.50) OR (NN1_distance <= {nn1_threshold:.15f})"
            ),
            "routed_rows": int(routed.sum()),
            "routing_rate_pct": float(100.0 * routed.mean()),
            "routed_normal": int((routed & normal_mask).sum()),
            "normal_capture_pct": float(
                100.0 * (routed & normal_mask).sum() / normal_mask.sum()
            ),
            "routed_attack": int((routed & attack_mask).sum()),
            "repeated_verifier_rows": repeated_rows,
            "duplicate_groups_crossing_folds": crossing_groups,
            "contradictory_duplicate_groups": contradictory_groups,
            "external_validation_used": False,
            "testing_csv_read": False,
        },
    )

    np.savez_compressed(
        out / "verifier_development_index.npz",
        source_train_index=source_idx.astype(np.int64),
        verifier_fold_id=fold_id[source_idx].astype(np.int8),
        route_reason=route_reason[source_idx].astype("U16"),
    )

    print(
        "Frozen router reproduced:",
        {
            "threshold": nn1_threshold,
            "routed": int(routed.sum()),
            "normal": int((routed & normal_mask).sum()),
            "attack": int((routed & attack_mask).sum()),
            "repeated_verifier_rows": repeated_rows,
        },
    )


if __name__ == "__main__":
    main()
