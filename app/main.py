"""TensorForge 2.0 ticket routing service.

Check order for prediction and job-submission endpoints (from the contract):
auth (401) -> content type (415) -> size (413) -> JSON parse (400) -> validation (422).
"""
import hmac
import json
import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import config
from .errors import ApiError, error_response
from .jobs import JobStore, QueueFull, Worker, status_body
from .model import Predictor
from .validation import validate_many, validate_single

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("service")


class State:
    predictor: Predictor | None = None
    load_error: str | None = None
    store: JobStore | None = None
    worker: Worker | None = None


S = State()


def _load_model():
    try:
        t = time.time()
        S.predictor = Predictor(config.MODEL_PATH)
        log.info("model %s loaded in %.1fs", S.predictor.version, time.time() - t)
        S.worker = Worker(S.store, lambda: S.predictor)
        S.worker.start()
    except Exception as e:  # keep serving /health as 503 instead of crashing
        S.load_error = str(e)
        log.exception("model failed to load")


@asynccontextmanager
async def lifespan(app):
    if config.api_key() is None:
        log.warning("API_KEY is not set: prediction and job endpoints will return 401")
    S.store = JobStore(config.JOB_DB_PATH)
    S.store.mark_interrupted()
    threading.Thread(target=_load_model, daemon=True, name="model-loader").start()
    yield
    if S.worker:
        S.worker.stop.set()


app = FastAPI(title="TensorForge 2.0 Ticket Router", lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)


# ---------------------------------------------------------------- middleware / handlers
@app.middleware("http")
async def request_id_echo(request: Request, call_next):
    rid = request.headers.get("x-request-id")
    response = await call_next(request)
    if rid and len(rid) <= 128 and rid.isprintable():
        response.headers["X-Request-ID"] = rid
    return response


@app.exception_handler(ApiError)
async def _api_error(request, exc: ApiError):
    return error_response(exc.status, exc.code, exc.message, exc.details, exc.headers)


@app.exception_handler(StarletteHTTPException)
async def _http_error(request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return error_response(404, "not_found", "Route not found.")
    if exc.status_code == 405:
        return error_response(405, "method_not_allowed", "Method not allowed for this route.")
    return error_response(exc.status_code, "http_error", str(exc.detail))


@app.exception_handler(RequestValidationError)
async def _validation_error(request, exc):
    return error_response(422, "validation_error", "Request failed validation.")


@app.exception_handler(Exception)
async def _unhandled(request, exc):
    log.exception("unhandled error")
    return error_response(500, "internal_error", "Internal server error.")


# ---------------------------------------------------------------- request guards
UNAUTH_HEADERS = {"WWW-Authenticate": "Bearer"}


def check_auth(request: Request):
    expected = config.api_key()
    candidates = []
    if (x := request.headers.get("x-api-key")) is not None:
        candidates.append(x.strip())
    auth = request.headers.get("authorization")
    if auth and auth[:7].lower() == "bearer ":
        candidates.append(auth[7:].strip())
    if not candidates:
        raise ApiError(401, "unauthorized", "Missing API key. Send X-API-Key or Authorization Bearer.", headers=UNAUTH_HEADERS)
    if expected is None:
        raise ApiError(401, "unauthorized", "Invalid API key.", headers=UNAUTH_HEADERS)
    exp = expected.strip().encode()
    ok = False
    for c in candidates:  # compare every candidate so timing does not leak which one matched
        ok |= hmac.compare_digest(c.encode(), exp)
    if not ok:
        raise ApiError(401, "unauthorized", "Invalid API key.", headers=UNAUTH_HEADERS)


async def read_json(request: Request, limit: int):
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype != "application/json":
        raise ApiError(415, "unsupported_media_type", "Content-Type must be application/json.")
    too_big = ApiError(413, "payload_too_large", "Request body exceeds the maximum allowed size.")
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > limit:
        raise too_big
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise too_big
    try:
        return json.loads(bytes(body).decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise ApiError(400, "malformed_json", "Request body is not valid JSON.")


def require_model() -> Predictor:
    if S.predictor is None:
        raise ApiError(503, "model_loading", "Model is not loaded yet. Retry shortly.", headers={"Retry-After": "5"})
    return S.predictor


def validation_failed(details):
    return ApiError(422, "validation_error", "Request failed validation.", details=details)


# ---------------------------------------------------------------- routes
@app.get("/health")
async def health():
    if S.predictor is None:
        return JSONResponse(status_code=503, content={"status": "loading", "model_version": None, "model_loaded": False})
    return {"status": "ok", "model_version": S.predictor.version, "model_loaded": True}


@app.post("/predict")
async def predict(request: Request):
    check_auth(request)
    body = await read_json(request, config.MAX_BODY_PREDICT)
    clean, errors = validate_single(body)
    if errors:
        raise validation_failed(errors)
    predictor = require_model()
    return (await run_in_threadpool(predictor.predict, [clean]))[0]


@app.post("/predict/batch")
async def predict_batch(request: Request):
    check_auth(request)
    t0 = time.perf_counter()
    body = await read_json(request, config.MAX_BODY_BATCH)
    tickets, errors = validate_many(body, config.MAX_BATCH_ITEMS)
    if errors:
        raise validation_failed(errors)
    predictor = require_model()
    preds = await run_in_threadpool(predictor.predict, tickets)
    return {"predictions": preds,
            "meta": {"count": len(preds), "model_version": predictor.version,
                     "processing_time_ms": int((time.perf_counter() - t0) * 1000)}}


@app.post("/batch/jobs")
async def submit_job(request: Request):
    check_auth(request)
    body = await read_json(request, config.MAX_BODY_JOBS)
    tickets, errors = await run_in_threadpool(validate_many, body, config.MAX_JOB_ITEMS)
    if errors:
        raise validation_failed(errors)
    idem = request.headers.get("idempotency-key")
    if idem is not None and (len(idem) > 128 or not idem.strip()):
        raise validation_failed([{"field": "Idempotency-Key", "issue": "must be 1 to 128 characters"}])
    predictor = require_model()
    try:
        row, _ = await run_in_threadpool(S.store.submit, tickets, predictor.version, idem)
    except QueueFull:
        raise ApiError(429, "too_many_jobs", "Job queue is full. Retry later.", headers={"Retry-After": "30"})
    return JSONResponse(status_code=202, content=status_body(row),
                        headers={"Location": f"/batch/jobs/{row['job_id']}", "Retry-After": str(config.POLL_RETRY_AFTER)})


def _job_or_error(job_id: str):
    row = S.store.get(job_id) if len(job_id) <= 128 else None
    if row is None:
        raise ApiError(404, "job_not_found", "No job with this id.")
    if S.store.is_expired(row):
        raise ApiError(410, "job_expired", "Job results have expired.")
    return row


@app.get("/batch/jobs/{job_id}")
async def job_status(job_id: str, request: Request):
    check_auth(request)
    row = await run_in_threadpool(_job_or_error, job_id)
    headers = {"Retry-After": str(config.POLL_RETRY_AFTER)} if row["status"] in ("queued", "running") else {}
    return JSONResponse(content=status_body(row), headers=headers)


@app.delete("/batch/jobs/{job_id}")
async def job_delete(job_id: str, request: Request):
    check_auth(request)
    deleted = len(job_id) <= 128 and await run_in_threadpool(S.store.delete, job_id)
    if not deleted:
        raise ApiError(404, "job_not_found", "No job with this id.")
    return Response(status_code=204)


def _int_param(request, name, default, lo, hi):
    raw = request.query_params.get(name)
    if raw is None:
        return default
    try:
        v = int(raw)
    except ValueError:
        v = None
    if v is None or v < lo or (hi is not None and v > hi):
        rng = f"between {lo} and {hi}" if hi is not None else f">= {lo}"
        raise validation_failed([{"field": name, "issue": f"must be an integer {rng}"}])
    return v


@app.get("/batch/jobs/{job_id}/results")
async def job_results(job_id: str, request: Request):
    check_auth(request)
    offset = _int_param(request, "offset", 0, 0, None)
    limit = _int_param(request, "limit", None, 1, config.MAX_JOB_ITEMS)
    row = await run_in_threadpool(_job_or_error, job_id)
    if row["status"] != "succeeded":
        raise ApiError(409, "job_not_ready",
                       f"Job status is '{row['status']}'. Results are available once it is 'succeeded'.")
    preds = await run_in_threadpool(S.store.results, job_id)
    if preds is None:
        raise ApiError(410, "job_expired", "Job results have expired.")
    total = len(preds)
    limit = limit if limit is not None else max(total, 1)
    page = preds[offset:offset + limit]
    nxt = offset + limit if offset + limit < total else None
    return {"job_id": job_id, "status": "succeeded", "total": total, "offset": offset, "limit": limit,
            "next_offset": nxt, "model_version": row["model_version"], "predictions": page}
