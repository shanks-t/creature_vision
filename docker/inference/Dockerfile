# Use the official lightweight Python image.
FROM python:3.10-slim

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED=True

ARG MODEL_VERSION
ENV MODEL_VERSION=${MODEL_VERSION}

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . . 

CMD ["python", "-m", "main"]
