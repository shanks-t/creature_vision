# Use the official lightweight Python image.
FROM python:3.10-slim

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

ARG MODEL_VERSION
ENV MODEL_VERSION=${MODEL_VERSION}

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the model with a fixed name
COPY ./model.keras /app/model.keras

COPY /app/main.py ./app/

# Run app.py when the container launches
CMD ["python", "app/main.py"]