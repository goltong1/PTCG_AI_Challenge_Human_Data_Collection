FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CABT_PUBLIC_MODE=1 \
    CABT_DATA_DIR=/data \
    PORT=8765

WORKDIR /srv/cabt

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY README.md VERSION.txt AI_FORMAT_GUIDE.md ./

RUN mkdir -p /data/records /data/submissions /data/user_data \
    && chmod -R 755 /srv/cabt /data

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8765')+'/api/health', timeout=4)"

CMD ["python", "app/server.py", "--host", "0.0.0.0", "--no-browser"]
