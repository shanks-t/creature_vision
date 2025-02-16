FROM python:3.10.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy and install requirements
COPY ../../src/training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ../../src/preprocessing src/preprocessing/
COPY ../../src/training src/training/
COPY docker/training/entrypoint.sh /

RUN chmod +x /entrypoint.sh

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import os; exit(0 if os.path.exists('/tmp/healthy') else 1)"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "src.training.main"]
