FROM python:3.12-slim

# System deps:
#   libspatialindex-dev  → rtree (geopandas spatial indexing)
#   libpango* + libharfbuzz + libcairo → WeasyPrint PDF rendering
#   curl                 → healthcheck / download fallback
RUN apt-get update && apt-get install -y --no-install-recommends \
        libspatialindex-dev \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libpangoft2-1.0-0 \
        libharfbuzz0b \
        libcairo2 \
        libffi-dev \
        libssl-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[dev]"

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "georeasoner.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
