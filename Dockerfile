FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/runtime \
    MODEL_PATH=/app/models/model.joblib

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/__init__.py src/__init__.py
COPY src/api/ src/api/
COPY src/model/ src/model/
COPY models/model.joblib models/model.joblib

RUN useradd -m svc && mkdir -p /app/runtime && chown -R svc /app/runtime
USER svc

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

# One process: job state is in SQLite on disk, inference runs off the request thread.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-server-header"]
