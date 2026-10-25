# InspectAI inspection API container (the AWS-hosted component).
# Serves the FastAPI app on port 8080 (App Runner / ECS default-friendly).
FROM python:3.11-slim

# OpenCV runtime needs a few shared libs even for headless use.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code.
COPY src ./src

ENV PORT=8080 \
    INSPECTAI_MAX_UPLOAD_BYTES=8388608 \
    INSPECTAI_MAX_AGENT_ITERS=4 \
    PYTHONUNBUFFERED=1

EXPOSE 8080

# Container health check hits the app's /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8080')+'/health')" || exit 1

CMD ["sh", "-c", "uvicorn src.api.routes:app --host 0.0.0.0 --port ${PORT}"]
