# Use the official lightweight Python image.
FROM python:3.10-slim

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

ARG MODEL_VERSION
ENV MODEL_VERSION=${MODEL_VERSION}

WORKDIR /app

COPY ../../src/training/requirements.txt .
RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY ../../src/training src/training/

ENV PYTHONPATH="/app/src:${PYTHONPATH}"
CMD ["python", "-m", "src.training.main"]