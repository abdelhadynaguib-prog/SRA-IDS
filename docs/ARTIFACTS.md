# Frozen artifacts

The trained binary models and large per-sample output archives are intentionally not stored in normal Git history.

They will be archived in the permanent Zenodo record. After downloading them, place the files under:

```text
artifacts/frozen/
```

Then verify them:

```bash
python scripts/verify_frozen_artifacts.py   --artifact-dir artifacts/frozen   --manifest manifests/frozen_artifact_sha256_expected.csv
```

The expected SHA-256 digests were copied from the permanently frozen final experiment.

After Zenodo deposition, update:
- `README.md`
- `CITATION.cff`
- the manuscript's Code and Artifact Availability statement

with the permanent DOI.
