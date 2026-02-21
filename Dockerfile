# ─── Stage 1: Test Runner ─────────────────────────────────────────────────────
FROM python:3.11-slim AS test

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install all deps including dev
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

# Run tests; build fails if tests fail
RUN pytest tests/unit -v --no-cov


# ─── Stage 2: Production Image ────────────────────────────────────────────────
FROM python:3.11-slim AS production

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
# Install only runtime deps (no dev extras)
RUN pip install --no-cache-dir -e .

COPY src/ src/
COPY prompts/ prompts/

# Create runtime directories
RUN mkdir -p data/raw data/checkpoints logs

# Run as non-root
RUN useradd -m -u 1000 pipeline
RUN chown -R pipeline:pipeline /app
USER pipeline

# Default entrypoint for weekly cron / Lambda handler
CMD ["python", "-m", "src.pipeline"]
