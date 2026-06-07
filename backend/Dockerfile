FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app/backend:/app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

COPY backend/requirements.txt ./requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY backend /app/backend
COPY engine /app/engine
COPY data /app/data

RUN mkdir -p /app/backend/uploads

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
