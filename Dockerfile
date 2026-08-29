# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=4000 \
    HOST=0.0.0.0 \
    SQLITE_PATH=/data/cache.db

WORKDIR /app

COPY pyproject.toml README.md ./
COPY cachellm ./cachellm
RUN pip install --no-cache-dir . && mkdir -p /data

VOLUME ["/data"]
EXPOSE 4000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:4000/health', timeout=4).status==200 else 1)"

CMD ["cachellm", "start"]
