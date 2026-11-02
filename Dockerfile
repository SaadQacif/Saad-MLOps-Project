FROM python:3.10-slim

# REST API image (src/api.py served by gunicorn on port 8000)
WORKDIR /app

# OpenCV runtime libraries; curl for the health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Locked dependencies, the same set CI and the other images install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Only what serving needs: code, trained models, and the training features used as
# the drift reference (not the image dataset)
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY models/bin/ ./models/bin/
COPY data/features/reference_features.csv ./data/features/reference_features.csv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health/status || exit 1

# one process with threads: the prediction log is guarded by an in-process lock
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--bind", "0.0.0.0:8000", "--timeout", "120", "src.api:app"]
