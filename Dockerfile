# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && pip install --no-cache-dir .

FROM python:3.11-slim AS runtime

RUN groupadd --gid 10001 phishnet \
    && useradd --uid 10001 --gid phishnet --create-home --shell /usr/sbin/nologin phishnet

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PHISHNET_MODEL_PATH=/app/models/phishnet-0.3.0.joblib

WORKDIR /app
COPY --chown=phishnet:phishnet models ./models

USER phishnet

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)" || exit 1

CMD ["uvicorn", "phishnet.api:app", "--host", "0.0.0.0", "--port", "8000"]
