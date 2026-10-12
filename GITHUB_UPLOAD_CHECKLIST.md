# GitHub upload checklist

1. Create a new **public** GitHub repository named `SRA-IDS`.
2. When creating it, do **not** add a README, `.gitignore`, or license from GitHub; this package already contains them.
3. Extract `SRA-IDS_GitHub_Source_v1.0.0.zip` locally.
4. Open the extracted `SRA-IDS-GitHub-v1.0.0` folder.
5. Upload the **contents of that folder** to the repository root.
6. Use commit message: `Initial public source release for SRA-IDS`.
7. Confirm that `README.md` renders correctly.
8. Confirm that GitHub recognizes the MIT license.
9. Confirm that `CITATION.cff` is in the repository root.
10. Add repository topics:
    - `intrusion-detection`
    - `iot-security`
    - `bot-iot`
    - `lightgbm`
    - `attention`
    - `cybersecurity`
    - `machine-learning`
11. Copy the final GitHub repository URL.
12. Edit `CITATION.cff` and replace `TO-BE-ADDED-AFTER-GITHUB-REPOSITORY-CREATION` with the repository URL.
13. Commit that change with message: `Add canonical repository URL`.
14. Do not upload BoT-IoT CSV files.
15. Do not upload trained binary artifacts into normal Git history; they will be archived on Zenodo.
16. After Zenodo deposition, add its DOI to `README.md` and `CITATION.cff`.
17. Create the GitHub release `v1.0.0` only after the repository text and Zenodo linkage are final.
