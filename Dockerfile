# syntax=docker/dockerfile:1

# ---- builder: resolve deps into an isolated prefix, no compiler needed (all wheels) ----
FROM python:3.13-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- runtime: only the installed packages + app code, non-root ----
FROM python:3.13-slim
WORKDIR /app

RUN groupadd -r app && useradd -r -m -d /home/app -g app app
ENV HOME=/home/app
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY --from=builder /install /usr/local
COPY logger.py .
COPY scripts/ ./scripts/

# runtime-only data, mount as volumes in prod (see .gitignore)
RUN mkdir -p data/log inputs outputs && chown -R app:app /app

USER app
EXPOSE 8501
VOLUME ["/app/data", "/app/inputs", "/app/outputs"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)" || exit 1

ENTRYPOINT ["streamlit", "run", "scripts/streamlit_app.py", \
    "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
