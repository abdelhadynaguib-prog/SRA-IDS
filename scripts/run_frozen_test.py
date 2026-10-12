from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sra_ids.inference import FrozenSRAIDS
from sra_ids.metrics import binary_metrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--test-csv",required=True)
    p.add_argument("--artifact-dir",required=True)
    p.add_argument("--output-dir",default="reproduced_test")
    args=p.parse_args()

    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(args.test_csv,low_memory=False)
    output=FrozenSRAIDS(args.artifact_dir).predict_frame(raw)
    np.savez_compressed(outdir/"reproduced_predictions_and_signals.npz",**output)

    if "attack" in raw.columns:
        y=pd.to_numeric(raw["attack"],errors="raise").to_numpy(dtype=np.uint8)
        metrics=binary_metrics(y,output["prediction"])
        (outdir/"reproduced_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
        print(json.dumps(metrics,indent=2))
    else:
        print("Predictions saved; target column 'attack' was not present.")

if __name__=="__main__":
    main()
