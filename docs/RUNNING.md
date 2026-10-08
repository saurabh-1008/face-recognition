# Running my baseline

## Setup (PowerShell, repository root)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Requirements record the versions already installed in this project's environment. All commands below use that environment. Model packs download on first use under `.venv/insightface/models/`; initialization may require internet. Existing local l/s packs have been copied there. If a download fails, read the report error and install the complete official pack under that directory before retrying.

## Register and verify

```powershell
.\.venv\Scripts\python.exe scripts/main.py --user-id user_001 --name ResearchParticipant --image data/images/IMG_20251213_193020.jpg --model buffalo_l
.\.venv\Scripts\python.exe scripts/verify.py --user-id user_001 --image data/images/IMG_20251213_193020.jpg --model buffalo_l
```

The database is created at `data/embeddings/users.json`. Users have a name and an embeddings dictionary keyed by model. Each entry contains a vector and UTC creation time. Use `--replace` to explicitly update an existing registration. Legacy NPY files are kept in `data/embeddings/` and are not imported because they lack model provenance. Face images are kept in `data/images/`. Paths to input images are relative to your shell; the database and model paths always resolve from the repository.

Verification is one-to-one against the supplied ID. Default threshold 0.5 is experimental and must be calibrated; similarity is not a probability. Exactly one detected face is required.

## Compare models

```powershell
.\.venv\Scripts\python.exe scripts/compare.py --pairs experiments/sanity_pairs.csv --output results/sanity_comparison.json
.\.venv\Scripts\python.exe scripts/compare.py --pairs experiments/pairs.template.csv --threshold 0.5
```

Replace the template image paths with real labelled pairs before running the second command. CSV image paths are relative to the CSV directory. Add `--models buffalo_l buffalo_s` to compare selected packs. Reports include scores, classification metrics, failure counts, ONNX size, timing and software versions. Per-model errors are saved and the command exits nonzero on failed models/pairs. Never interpret missing metrics as success.

See [research protocol](../reports/RESEARCH_PROTOCOL.md) for evaluation design and [structure.txt](../reports/PROJECT_STRUCTURE.md) for the repo layout. Raw images and the working user database are excluded from Git; source and aggregate results are intended for review.

## Generate comparison graphs

```powershell
.\.venv\Scripts\python.exe scripts/plot_results.py
```

Graphs are saved as PNG and SVG in `results/graphs/`. Use `--input results/comparison.json --output results/graphs/evaluation` for another comparison report.

## Evaluate all available images

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_images.py
```

This evaluates every image in `data/images/` against all four models. Per-image outcomes, pairwise scores and a success/failure graph are saved under `results/`; the report is `reports/IMAGE_EVALUATION.md`. It does not change the user database.

For identity evaluation, fill `experiments/image_labels.csv` with a consistent person ID for each image of the same person, then run:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_images.py --labels experiments/image_labels.csv
```

Blank person IDs remain unknown. A detection/extraction success is distinct from a correct identity match; unlabelled pairs have no accuracy result.
