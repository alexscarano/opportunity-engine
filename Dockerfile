# syntax=docker/dockerfile:1

# ---- web: build the SPA ----
FROM oven/bun:1.3.14 AS web
WORKDIR /web
COPY web/package.json web/bun.lock ./
RUN bun install --frozen-lockfile
COPY web/ ./
RUN bun run build

# ---- builder: resolve python deps from uv.lock into /app/.venv ----
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv
COPY pyproject.toml uv.lock .python-version ./
# strip vendored test suites from deps (~55MB dead weight, e.g. scipy/pandas/statsmodels tests)
RUN uv sync --frozen --no-dev --no-install-project \
    && find /app/.venv -depth -type d \( -name "tests" -o -name "test" \) -exec rm -rf {} +

# ---- runtime: api (default), worker job and streamlit share this image ----
FROM python:3.13-slim
WORKDIR /app

RUN groupadd -r app && useradd -r -m -d /home/app -g app app
ENV HOME=/home/app \
    PATH=/app/.venv/bin:$PATH \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY --from=builder /app/.venv /app/.venv
COPY logger.py .
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY Logos/ ./Logos/
COPY --from=web /web/dist ./web/dist

# runtime-only data for streamlit, mount as volumes (see .gitignore)
RUN mkdir -p data/log inputs outputs && chown -R app:app /app

USER app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8080\")}/api/health', timeout=3)" || exit 1

# shell form so Cloud Run's $PORT expands. Worker: `python -m api.worker`.
# Streamlit: `streamlit run scripts/streamlit_app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true`
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
