from __future__ import annotations
import argparse,json,numpy as np
from sra_ids.metrics import binary_metrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--npz",required=True)
    args=p.parse_args()
    data=np.load(args.npz)
    required={"y_true","hybrid_prediction"}
    missing=required-set(data.files)
    if missing:
        raise KeyError(f"Missing arrays: {sorted(missing)}; available: {sorted(data.files)}")
    print(json.dumps(binary_metrics(data["y_true"],data["hybrid_prediction"]),indent=2))

if __name__=="__main__":
    main()
