FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MEMORY_DB_PATH=/app/data/memory.db

WORKDIR /app
COPY requirements*.txt ./
ARG INSTALL_LOCAL_MODELS=false
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_LOCAL_MODELS" = "true" ]; then \
         apt-get update \
         && apt-get install -y --no-install-recommends build-essential cmake libgomp1 \
         && pip install --no-cache-dir -r requirements-local.txt \
         && apt-get purge -y --auto-remove build-essential cmake \
         && rm -rf /var/lib/apt/lists/*; \
       fi

RUN useradd --create-home appuser && mkdir -p /app/data \
    && chown appuser:appuser /app/data
COPY --chown=appuser:appuser . .
USER appuser
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "600", "app:app"]
