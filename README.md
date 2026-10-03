# TensorForge 2.0: RideEat Ticket Router

Classifies RideEat support tickets (English, Sinhala, Tamil, Singlish, Tanglish, mixed) into a primary
category, an optional secondary category and an urgency flag, and routes them to a team.
The service follows `tensorforge-phase2-openapi-v2.yaml` exactly.

## Run with Docker

```powershell
docker build -t tensorforge .
docker run -p 8000:8000 -e API_KEY=<key> tensorforge
```

No internet is needed at runtime. The model is baked into the image.

## Run locally (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
$env:API_KEY = "dev-key"
uvicorn app.main:app --port 8000
```

## Test

```powershell
pytest -q
```

52 contract tests. Every response is validated against the official JSON Schemas in `schemas/`.
They cover auth order, 400/401/404/405/409/410/413/415/422/429 paths, batch ordering and
atomic validation, async jobs (paging, idempotency, queue limit, expiry, restart recovery) and
determinism between `/predict`, `/predict/batch` and jobs.

## Train

Put `train.csv` and `validation.csv` from the organisers in `data/` (not committed), then:

```powershell
python training/train_baseline.py
```

This writes `artifacts/model.joblib` and `artifacts/metrics.json`.

## Layout

| Path | What it does |
|---|---|
| `app/main.py` | Routes and the check order: auth, content type, size, JSON, validation |
| `app/validation.py` | Field-level validation with `index` for batch items |
| `app/model.py` | Model backends and the rules layer (team map, spam rules, secondary != primary) |
| `app/jobs.py` | SQLite job store and background worker |
| `app/textprep.py` | Text preparation shared by training and serving |
| `training/` | Training scripts |
| `artifacts/` | Versioned model files |

## Model version

`model_version` is `<tag>+<first 8 hex of the artifact's SHA-256>`, so it always points at an exact
file in `artifacts/`. It is identical across `/health`, `/predict`, `/predict/batch` and jobs.

## needs_human_review

Set to `true` when the calibrated confidence of the primary category is below 0.5.
Confidence is calibrated with temperature scaling fitted on the validation set.

## Async jobs

Jobs are stored in SQLite under `DATA_DIR` (default `/app/runtime`). Mount a volume there on the
hosted server so jobs survive restarts. A job that was running when the service stopped is marked
`failed` with `error.code = interrupted` on the next start. Results are kept for 24 hours
(spec minimum is 6), then return `410`.

| Setting | Default |
|---|---|
| `MAX_ACTIVE_JOBS` | 4 (1 running + 3 queued) |
| `JOB_RETENTION_HOURS` | 24 |
| `JOB_CHUNK_SIZE` | 32 |

## Current model

Baseline `tfidf-lr-0.1` (char + word TF-IDF, three logistic regression heads). Placeholder until the
multilingual transformer is trained. See `artifacts/metrics.json`.
