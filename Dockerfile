# syntax=docker/dockerfile:1
#
# Serves the API and the dashboard from one image. The pipelines run as one-off
# commands against the same image (see docker-compose.yml), so training and
# serving share a single dependency set and cannot drift apart.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# LightGBM needs libgomp at runtime; it is not in the slim image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

# Dependencies first, so a source edit does not invalidate the install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY tests/ ./tests/
COPY pytest.ini .

# Written to at runtime by the pipelines.
RUN mkdir -p data models reports/figures

# Run as a non-root user; the writable directories are handed over explicitly.
RUN useradd --create-home --uid 10001 aqi \
 && chown -R aqi:aqi /app/data /app/models /app/reports
USER aqi

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
