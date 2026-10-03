"""Async batch jobs. State lives in SQLite on disk so every worker/replica sees the same jobs
and a restart turns running jobs into failed/interrupted instead of losing them."""
import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone

from . import config

log = logging.getLogger("jobs")

ACTIVE = ("queued", "running")


def now_utc():
    return datetime.now(timezone.utc)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt else None


def parse(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) if s else None


class JobStore:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with closing(self._conn()) as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, status TEXT NOT NULL, total INTEGER NOT NULL,
                processed INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
                started_at TEXT, finished_at TEXT, expires_at TEXT, model_version TEXT,
                error_code TEXT, error_message TEXT, idempotency_key TEXT UNIQUE,
                tickets TEXT, results TEXT, purged INTEGER NOT NULL DEFAULT 0)""")

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        return c

    # ---------- startup ----------
    def mark_interrupted(self):
        t = now_utc()
        with closing(self._conn()) as c:
            n = c.execute(
                "UPDATE jobs SET status='failed', error_code='interrupted', "
                "error_message='Service restarted while the job was running.', "
                "finished_at=?, expires_at=?, tickets=NULL WHERE status='running'",
                (iso(t), iso(t + timedelta(hours=config.JOB_RETENTION_HOURS))),
            ).rowcount
        if n:
            log.warning("marked %d running job(s) as interrupted", n)

    # ---------- API side ----------
    def submit(self, tickets, model_version, idempotency_key=None):
        """Return (row, created). Raises QueueFull when too many active jobs."""
        c = self._conn()
        try:
            c.execute("BEGIN IMMEDIATE")
            if idempotency_key:
                row = c.execute("SELECT * FROM jobs WHERE idempotency_key=?", (idempotency_key,)).fetchone()
                if row:
                    c.execute("COMMIT")
                    return row, False
            active = c.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if active >= config.MAX_ACTIVE_JOBS:
                c.execute("ROLLBACK")
                raise QueueFull()
            job_id = str(uuid.uuid4())
            c.execute(
                "INSERT INTO jobs (job_id,status,total,processed,created_at,model_version,idempotency_key,tickets) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (job_id, "queued", len(tickets), 0, iso(now_utc()), model_version, idempotency_key,
                 json.dumps(tickets, ensure_ascii=False)),
            )
            c.execute("COMMIT")
            return c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone(), True
        except QueueFull:
            raise
        except Exception:
            try:
                c.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            c.close()

    def get(self, job_id):
        with closing(self._conn()) as c:
            return c.execute(
                "SELECT job_id,status,total,processed,created_at,started_at,finished_at,expires_at,"
                "model_version,error_code,error_message,purged FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()

    def results(self, job_id):
        with closing(self._conn()) as c:
            row = c.execute("SELECT results FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return json.loads(row["results"]) if row and row["results"] else None

    def delete(self, job_id):
        with closing(self._conn()) as c:
            return c.execute("DELETE FROM jobs WHERE job_id=?", (job_id,)).rowcount > 0

    @staticmethod
    def is_expired(row):
        exp = parse(row["expires_at"])
        return bool(row["purged"]) or (exp is not None and now_utc() >= exp)

    def purge_expired(self):
        t = now_utc()
        with closing(self._conn()) as c:
            c.execute("UPDATE jobs SET results=NULL, tickets=NULL, purged=1 WHERE purged=0 AND expires_at IS NOT NULL AND expires_at<=?", (iso(t),))
            # forget ids a week after expiry
            c.execute("DELETE FROM jobs WHERE purged=1 AND expires_at<=?", (iso(t - timedelta(days=7)),))

    # ---------- worker side ----------
    def claim_next(self):
        c = self._conn()
        try:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT job_id, tickets FROM jobs WHERE status='queued' ORDER BY created_at, rowid LIMIT 1").fetchone()
            if not row:
                c.execute("COMMIT")
                return None
            c.execute("UPDATE jobs SET status='running', started_at=? WHERE job_id=?", (iso(now_utc()), row["job_id"]))
            c.execute("COMMIT")
            return row["job_id"], json.loads(row["tickets"])
        except Exception:
            try:
                c.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            c.close()

    def progress(self, job_id, processed):
        """Returns False if the job was deleted (cancelled) meanwhile."""
        with closing(self._conn()) as c:
            return c.execute("UPDATE jobs SET processed=? WHERE job_id=? AND status='running'", (processed, job_id)).rowcount > 0

    def finish(self, job_id, results=None, error=None):
        t = now_utc()
        exp = iso(t + timedelta(hours=config.JOB_RETENTION_HOURS))
        with closing(self._conn()) as c:
            if error:
                c.execute("UPDATE jobs SET status='failed', error_code=?, error_message=?, finished_at=?, expires_at=?, tickets=NULL "
                          "WHERE job_id=? AND status='running'", (error[0], error[1], iso(t), exp, job_id))
            else:
                c.execute("UPDATE jobs SET status='succeeded', processed=total, results=?, finished_at=?, expires_at=?, tickets=NULL "
                          "WHERE job_id=? AND status='running'", (json.dumps(results, ensure_ascii=False), iso(t), exp, job_id))


class QueueFull(Exception):
    pass


def status_body(row):
    error = None
    if row["status"] == "failed":
        error = {"code": row["error_code"] or "internal_error", "message": row["error_message"] or "Job failed."}
    return {
        "job_id": row["job_id"], "status": row["status"], "total": row["total"], "processed": row["processed"],
        "created_at": row["created_at"], "started_at": row["started_at"], "finished_at": row["finished_at"],
        "expires_at": row["expires_at"], "model_version": row["model_version"], "error": error,
    }


class Worker(threading.Thread):
    """Single background thread. Never runs on a request thread."""

    def __init__(self, store: JobStore, get_predictor):
        super().__init__(daemon=True, name="job-worker")
        self.store = store
        self.get_predictor = get_predictor
        self.stop = threading.Event()

    def run(self):
        last_purge = 0.0
        while not self.stop.is_set():
            try:
                if time.time() - last_purge > 60:
                    self.store.purge_expired()
                    last_purge = time.time()
                claimed = self.store.claim_next()
                if not claimed:
                    self.stop.wait(0.5)
                    continue
                self._run_job(*claimed)
            except Exception:
                log.exception("worker loop error")
                self.stop.wait(1)

    def _run_job(self, job_id, tickets):
        predictor = self.get_predictor()
        results = []
        try:
            for start in range(0, len(tickets), config.JOB_CHUNK_SIZE):
                chunk = tickets[start:start + config.JOB_CHUNK_SIZE]
                results.extend(predictor.predict(chunk))
                if not self.store.progress(job_id, len(results)):
                    log.info("job %s cancelled", job_id)
                    return
                time.sleep(0)  # let request threads breathe
            self.store.finish(job_id, results=results)
        except Exception:
            log.exception("job %s failed", job_id)
            self.store.finish(job_id, error=("internal_error", "Job failed during processing."))
