# Use the official lightweight Python image.
FROM python:3.10-slim

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

ARG MODEL_VERSION
ENV MODEL_VERSION=${MODEL_VERSION}

WORKDIR /app

COPY ../../src/inference/requirements.txt .
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the model with a fixed name
COPY ../../models/m_net_v3.0.keras /app/model.keras

COPY ../../src/inference/main.py ./app/

# Run app.py when the container launches
CMD ["python", "app/main.py"]