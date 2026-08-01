FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CABT_PUBLIC_MODE=1 \
    CABT_ENABLE_ONLINE_MATCHING=1 \
    CABT_ENABLE_RESULT_SUBMISSION=1 \
    CABT_DATA_DIR=/data \
    CABT_COOKIE_SECURE=1 \
    CABT_MAX_ACTIVE_WORKERS=1 \
    CABT_MAX_SESSIONS=1 \
    CABT_MAX_PVP_MATCHES=1 \
    CABT_MAX_PVP_WAITERS=12 \
    CABT_SESSION_IDLE_SECONDS=1200 \
    CABT_PVP_QUEUE_TIMEOUT=600 \
    CABT_PVP_MATCH_IDLE_SECONDS=3600

WORKDIR /srv/cabt

RUN apt-get update \
    && apt-get install -y --no-install-recommends tini libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY README.md VERSION.txt AI_FORMAT_GUIDE.md DEPLOY_RAILWAY.md ./

RUN mkdir -p /data/records /data/submissions /data/user_data \
    && chmod -R 755 /srv/cabt /data

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8765')+'/api/health', timeout=4)"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "app/server.py", "--host", "0.0.0.0", "--no-browser"]
