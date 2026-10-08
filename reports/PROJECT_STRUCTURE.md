# My project structure

I keep the review reports together in `reports/`. The top-level README provides the starting point for review.

```text
cdac/
|-- README.md                 Start here: findings and report links
|-- reports/                  Evaluation, limitations and methodology
|   |-- README.md             Report index and suggested reading order
|   |-- IMAGE_EVALUATION.md    Per-image and same-person pair outcomes
|   |-- ISSUES.md              Observed model limitations
|   |-- RESEARCH_PROTOCOL.md   Evaluation approach and remaining work
|   |-- SANITY_RESULTS.md      Identical-image checks and initial measurements
|   `-- PROJECT_STRUCTURE.md   This guide
|-- results/                  Supporting JSON, CSV and graph evidence
|-- scripts/                  Registration, verification and evaluation code
|-- experiments/              Image labels and pair CSV files
|-- data/
|   |-- images/               Local input images
|   `-- embeddings/           JSON database and legacy embeddings
|-- docs/RUNNING.md            Setup and execution commands
|-- baseline/                 Original implementation and archived notes
|-- requirements.txt          Python dependencies
`-- .venv/                    Local environment and downloaded models
```

My current pipeline is:

```text
Image -> face detection and alignment -> normalized embedding
Registration -> model-specific embedding in JSON
Verification -> same model -> cosine similarity -> threshold decision
```

The archived liveness diagram is a design note. Liveness detection is not implemented in this baseline.
