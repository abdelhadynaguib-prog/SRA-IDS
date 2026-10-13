from __future__ import annotations

import argparse
from decimal import Decimal, getcontext
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from common import save_json

# STEP 26T5 used arbitrary-precision exact binomial probabilities so that
# extremely small McNemar p-values are not rounded/underflowed to zero.
getcontext().prec = 200
ALPHA = Decimal("0.05")

COMPARATOR_KEYS = {
    "Random Forest": "rf_prediction",
    "XGBoost": "xgb_prediction",
    "LightGBM": "lgb_prediction",
    "DNN": "dnn_prediction",
    "LSTM": "lstm_prediction",
    "CNN-LSTM": "cnn_lstm_prediction",
    "Attention-CNN-LSTM": "attention_prediction",
}
EXPECTED_FULL_ERRORS = {
    "Frozen SRA-IDS": 9,
    "Random Forest": 5,
    "XGBoost": 4,
    "LightGBM": 6,
    "DNN": 621,
    "LSTM": 296,
    "CNN-LSTM": 752,
    "Attention-CNN-LSTM": 1848,
}
EXPECTED_SIGNIFICANT = {"DNN", "LSTM", "CNN-LSTM", "Attention-CNN-LSTM"}


def exact_mcnemar_decimal(b: int, c: int) -> Decimal:
    b = int(b)
    c = int(c)
    n = b + c
    if n == 0:
        return Decimal(1)
    r = min(b, c)
    if 2 * r >= n:
        return Decimal(1)
    numerator = sum(comb(n, k) for k in range(r + 1))
    denominator = 1 << n
    p_value = Decimal(2) * Decimal(numerator) / Decimal(denominator)
    return min(Decimal(1), p_value)


def holm_decimal(p_values: list[Decimal]) -> tuple[list[Decimal], list[int], list[Decimal]]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted: list[Decimal | None] = [None] * m
    ranks: list[int | None] = [None] * m
    critical: list[Decimal | None] = [None] * m
    running_max = Decimal(0)

    for sorted_position, original_index in enumerate(order):
        multiplier = Decimal(m - sorted_position)
        candidate = min(Decimal(1), p_values[original_index] * multiplier)
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
        ranks[original_index] = sorted_position + 1
        critical[original_index] = ALPHA / multiplier

    return (
        [x if x is not None else Decimal(1) for x in adjusted],
        [int(x) if x is not None else 0 for x in ranks],
        [x if x is not None else ALPHA for x in critical],
    )


def dstr(value: Decimal, digits: int = 24) -> str:
    if value == 0:
        return "0"
    return format(value, f".{digits}E")


def choose_key(archive, candidates: list[str], label: str) -> str:
    for key in candidates:
        if key in archive.files:
            return key
    raise KeyError(f"Could not find {label}; tried: {candidates}")


def paired_family(
    y_true: np.ndarray,
    hybrid_pred: np.ndarray,
    comparator_predictions: dict[str, np.ndarray],
    mask: np.ndarray,
    family_name: str,
) -> pd.DataFrame:
    y = y_true[mask]
    h = hybrid_pred[mask]
    hybrid_correct = h == y
    rows = []
    p_values: list[Decimal] = []

    for name, pred_all in comparator_predictions.items():
        pred = pred_all[mask]
        comp_correct = pred == y
        b = int(np.sum(hybrid_correct & ~comp_correct))
        c = int(np.sum(~hybrid_correct & comp_correct))
        both_wrong = int(np.sum(~hybrid_correct & ~comp_correct))
        both_correct = int(np.sum(hybrid_correct & comp_correct))
        hybrid_errors = int(np.sum(~hybrid_correct))
        comparator_errors = int(np.sum(~comp_correct))
        p = exact_mcnemar_decimal(b, c)
        p_values.append(p)
        rows.append({
            "Family": family_name,
            "Comparator": name,
            "Rows": int(mask.sum()),
            "Hybrid_Correct_Comparator_Wrong_b": b,
            "Hybrid_Wrong_Comparator_Correct_c": c,
            "Both_Correct": both_correct,
            "Both_Wrong": both_wrong,
            "Hybrid_Errors": hybrid_errors,
            "Comparator_Errors": comparator_errors,
            "Comparator_Errors_minus_Hybrid": comparator_errors - hybrid_errors,
            "McNemar_Exact_P": p,
        })

    adjusted, ranks, critical = holm_decimal(p_values)
    for i, row in enumerate(rows):
        row["Holm_Rank"] = ranks[i]
        row["Holm_Critical_Alpha"] = critical[i]
        row["Holm_Adjusted_P"] = adjusted[i]
        significant = bool(adjusted[i] <= ALPHA)
        row["Holm_Significant_0.05"] = significant
        b = row["Hybrid_Correct_Comparator_Wrong_b"]
        c = row["Hybrid_Wrong_Comparator_Correct_c"]
        if not significant:
            interpretation = "No statistically significant paired difference after Holm correction"
        elif b > c:
            interpretation = "Statistically significant paired error difference in favor of SRA-IDS"
        elif c > b:
            interpretation = "Statistically significant paired error difference in favor of comparator"
        else:
            interpretation = "Statistically significant status with equal discordant counts"
        row["Statistical_Interpretation"] = interpretation

    # Serialize Decimal columns as authoritative scientific-notation strings.
    out_rows = []
    for row in rows:
        row = dict(row)
        row["McNemar_Exact_P"] = dstr(row["McNemar_Exact_P"])
        row["Holm_Critical_Alpha"] = dstr(row["Holm_Critical_Alpha"])
        row["Holm_Adjusted_P"] = dstr(row["Holm_Adjusted_P"])
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce STEP 26T5 high-precision paired McNemar + Holm analysis.")
    ap.add_argument("--sra-npz", required=True, help="Frozen SRA prediction archive")
    ap.add_argument("--comparator-npz", required=True, help="STEP-26S comparator prediction archive")
    ap.add_argument("--overlap-npz", required=True, help="STEP-26T4 exact overlap mask archive")
    ap.add_argument("--output-dir", default="reproduction_outputs/12_paired_statistics")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with np.load(args.sra_npz, allow_pickle=False) as sra:
        y_key = choose_key(sra, ["y_true", "y_test"], "SRA y_true")
        pred_key = choose_key(
            sra, ["hybrid_prediction", "sra_prediction", "prediction", "y_pred"],
            "SRA prediction",
        )
        y_true = sra[y_key].astype(np.uint8, copy=True)
        hybrid_pred = sra[pred_key].astype(np.uint8, copy=True)

    comparator_predictions: dict[str, np.ndarray] = {}
    with np.load(args.comparator_npz, allow_pickle=False) as comp:
        if "y_true" not in comp.files:
            raise KeyError("Comparator archive missing y_true.")
        comp_y = comp["y_true"].astype(np.uint8, copy=True)
        if not np.array_equal(comp_y, y_true):
            raise RuntimeError("SRA and comparator y_true arrays are not exactly aligned.")
        for name, key in COMPARATOR_KEYS.items():
            if key not in comp.files:
                raise KeyError(f"Comparator archive missing {key}.")
            comparator_predictions[name] = comp[key].astype(np.uint8, copy=True)

    with np.load(args.overlap_npz, allow_pickle=False) as overlap:
        key = choose_key(overlap, ["common_protocol_seen_mask", "exact_seen"], "exact seen mask")
        seen = overlap[key].astype(bool, copy=True)

    if len(y_true) != 733_705 or len(hybrid_pred) != len(y_true) or len(seen) != len(y_true):
        raise RuntimeError("Unexpected frozen test-array lengths.")
    if int(seen.sum()) != 3_484:
        raise RuntimeError(f"Expected 3,484 exact-seen rows, observed {int(seen.sum())}.")
    unseen = ~seen

    # Frozen full-test error-count guardrail.
    observed_hybrid_errors = int(np.sum(hybrid_pred != y_true))
    if observed_hybrid_errors != EXPECTED_FULL_ERRORS["Frozen SRA-IDS"]:
        raise RuntimeError(f"Frozen SRA error count mismatch: {observed_hybrid_errors}")
    for name, pred in comparator_predictions.items():
        errors = int(np.sum(pred != y_true))
        if errors != EXPECTED_FULL_ERRORS[name]:
            raise RuntimeError(f"{name} frozen error count mismatch: {errors} != {EXPECTED_FULL_ERRORS[name]}")

    full_df = paired_family(
        y_true, hybrid_pred, comparator_predictions,
        np.ones(len(y_true), dtype=bool), "Full Test",
    )
    unseen_df = paired_family(
        y_true, hybrid_pred, comparator_predictions,
        unseen, "Common-Protocol Exact-Unseen",
    )

    full_sig = set(full_df.loc[full_df["Holm_Significant_0.05"], "Comparator"].tolist())
    unseen_sig = set(unseen_df.loc[unseen_df["Holm_Significant_0.05"], "Comparator"].tolist())
    if full_sig != EXPECTED_SIGNIFICANT:
        raise RuntimeError(f"Unexpected full-test Holm significance set: {sorted(full_sig)}")
    if unseen_sig != EXPECTED_SIGNIFICANT:
        raise RuntimeError(f"Unexpected exact-unseen Holm significance set: {sorted(unseen_sig)}")

    # The frozen SRA's nine errors must all remain in exact-unseen traffic.
    hybrid_seen_errors = int(np.sum((hybrid_pred != y_true) & seen))
    hybrid_unseen_errors = int(np.sum((hybrid_pred != y_true) & unseen))
    if hybrid_seen_errors != 0 or hybrid_unseen_errors != 9:
        raise RuntimeError(
            f"Unexpected SRA error localization: seen={hybrid_seen_errors}, unseen={hybrid_unseen_errors}"
        )

    full_path = out / "full_test_high_precision_mcnemar_holm.csv"
    unseen_path = out / "common_protocol_unseen_high_precision_mcnemar_holm.csv"
    combined_path = out / "exact_mcnemar_holm_full_and_unseen.csv"
    stability_path = out / "full_vs_unseen_statistical_stability.csv"
    full_df.to_csv(full_path, index=False)
    unseen_df.to_csv(unseen_path, index=False)
    pd.concat([full_df, unseen_df], ignore_index=True).to_csv(combined_path, index=False)

    stability_rows = []
    for name in COMPARATOR_KEYS:
        fr = full_df.loc[full_df["Comparator"] == name].iloc[0]
        ur = unseen_df.loc[unseen_df["Comparator"] == name].iloc[0]
        stability_rows.append({
            "Comparator": name,
            "Full_Holm_Significant": bool(fr["Holm_Significant_0.05"]),
            "Exact_Unseen_Holm_Significant": bool(ur["Holm_Significant_0.05"]),
            "Significance_Status_Preserved": bool(
                fr["Holm_Significant_0.05"] == ur["Holm_Significant_0.05"]
            ),
            "Full_McNemar_Exact_P": fr["McNemar_Exact_P"],
            "Full_Holm_Adjusted_P": fr["Holm_Adjusted_P"],
            "Unseen_McNemar_Exact_P": ur["McNemar_Exact_P"],
            "Unseen_Holm_Adjusted_P": ur["Holm_Adjusted_P"],
        })
    stability_df = pd.DataFrame(stability_rows)
    if not bool(stability_df["Significance_Status_Preserved"].all()):
        raise RuntimeError("Full-vs-unseen significance status was not preserved for every comparator.")
    stability_df.to_csv(stability_path, index=False)

    save_json(out / "step26T5_high_precision_statistical_summary.json", {
        "primary_family": "Full Test — seven SRA-vs-comparator paired comparisons",
        "secondary_robustness_family": "Common-protocol Exact-Unseen — seven SRA-vs-comparator paired comparisons",
        "correction": "Holm-Bonferroni separately within each seven-test family",
        "family_wise_alpha": 0.05,
        "full_rows": 733_705,
        "common_protocol_unseen_rows": 730_221,
        "full_significant_comparators": sorted(full_sig),
        "unseen_significant_comparators": sorted(unseen_sig),
        "all_significance_statuses_preserved": True,
        "numerical_underflow_avoided": True,
        "arithmetic": "Python integer combinations + Decimal exact binomial probabilities",
        "model_training_performed": False,
        "model_inference_performed": False,
        "threshold_changed": False,
        "model_selection_performed": False,
        "saved_predictions_only": True,
    })

    print(full_df.to_string(index=False))
    print(unseen_df.to_string(index=False))
    print("STEP-26T5 high-precision paired statistics reproduced successfully.")


if __name__ == "__main__":
    main()
