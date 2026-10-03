import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "artifacts" / "model.joblib"))
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "runtime"))
JOB_DB_PATH = Path(os.environ.get("JOB_DB_PATH", DATA_DIR / "jobs.sqlite3"))

MAX_BODY_PREDICT = 1 * 1024 * 1024
MAX_BODY_BATCH = 5 * 1024 * 1024
MAX_BODY_JOBS = 25 * 1024 * 1024

MAX_BATCH_ITEMS = 100
MAX_JOB_ITEMS = 5000
MAX_TEXT_CHARS = 10000
MAX_SUBJECT_CHARS = 500
MAX_TICKET_ID_CHARS = 64

MAX_ACTIVE_JOBS = int(os.environ.get("MAX_ACTIVE_JOBS", "4"))  # 1 running + 3 queued
JOB_RETENTION_HOURS = float(os.environ.get("JOB_RETENTION_HOURS", "24"))  # spec minimum is 6
JOB_CHUNK_SIZE = int(os.environ.get("JOB_CHUNK_SIZE", "32"))
POLL_RETRY_AFTER = 3


def api_key() -> str | None:
    """Read the expected key at call time. Empty or missing means the service refuses."""
    key = os.environ.get("API_KEY", "")
    return key if key.strip() else None
