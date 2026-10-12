from __future__ import annotations
import argparse,csv,hashlib
from pathlib import Path

def sha256_file(path,block_size=1024*1024):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            block=f.read(block_size)
            if not block: break
            h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--artifact-dir",required=True)
    p.add_argument("--manifest",required=True)
    args=p.parse_args()
    artifact_dir=Path(args.artifact_dir)
    failures=0
    with open(args.manifest,newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        path=artifact_dir/row["filename"]
        if not path.exists():
            print(f"MISSING  {row['filename']}"); failures+=1; continue
        if row.get("size_bytes","").strip() and int(row["size_bytes"])!=path.stat().st_size:
            print(f"SIZEFAIL {row['filename']}"); failures+=1; continue
        observed=sha256_file(path)
        if observed.lower()!=row["sha256"].lower():
            print(f"HASHFAIL {row['filename']}"); failures+=1
        else:
            print(f"PASS     {row['filename']}")
    if failures:
        raise SystemExit(f"{failures} artifact verification failure(s).")
    print("All listed frozen artifacts passed SHA-256 verification.")

if __name__=="__main__":
    main()
