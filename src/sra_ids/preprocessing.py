from __future__ import annotations
import numpy as np
import pandas as pd
from .constants import FEATURES, NUMERIC_FEATURES, PROTO_MAP

def parse_port_column(series: pd.Series) -> tuple[pd.Series, int]:
    """Convert decimal and 0x-prefixed hexadecimal ports; retain -1 ARP values."""
    text = series.astype(str).str.strip()
    lower = text.str.lower()
    hex_mask = lower.str.startswith("0x")
    result = pd.to_numeric(text.where(~hex_mask), errors="coerce")
    hex_count = int(hex_mask.sum())
    if hex_count:
        hex_values = np.fromiter(
            (int(value, 16) for value in lower[hex_mask]),
            dtype=np.int64,
            count=hex_count,
        )
        result.loc[hex_mask] = hex_values
    if result.isna().any():
        bad = text[result.isna()].head(10).tolist()
        raise ValueError(f"Unparseable port values detected: {bad}")
    return result.astype(np.float32), hex_count

def clean_bot_iot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = frame[FEATURES].copy()
    proto_text = out["proto"].astype(str).str.strip().str.lower()
    unknown = sorted(set(proto_text.dropna().unique()) - set(PROTO_MAP))
    if unknown:
        raise ValueError(f"Unknown protocol values: {unknown}")
    out["proto"] = proto_text.map(PROTO_MAP).astype(np.float32)
    out["sport"], _ = parse_port_column(out["sport"])
    out["dport"], _ = parse_port_column(out["dport"])
    for col in NUMERIC_FEATURES[2:]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float32)
    if out.isna().any().any():
        bad_cols = out.columns[out.isna().any()].tolist()
        raise ValueError(f"NaN values after cleaning in: {bad_cols}")
    values = out[FEATURES].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite values detected after cleaning.")
    return out

def make_routing_representation(frame: pd.DataFrame, scaler) -> np.ndarray:
    numeric = scaler.transform(frame[NUMERIC_FEATURES]).astype(np.float32)
    proto = frame["proto"].to_numpy(dtype=np.int32)
    if np.any((proto < 0) | (proto > 4)):
        raise ValueError("Unexpected protocol code.")
    proto_onehot = np.zeros((len(frame), 5), dtype=np.float32)
    proto_onehot[np.arange(len(frame)), proto] = 1.0
    return np.concatenate([numeric, proto_onehot], axis=1).astype(np.float32)
