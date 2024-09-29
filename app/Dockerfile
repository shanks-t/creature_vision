# Use the official lightweight Python image.
FROM python:3.10-slim

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

WORKDIR /app
# Copy local code to the container image.
COPY app.py ./app/
COPY mobile_net_v3_small.keras ./mobile_net_v3_small.keras

COPY requirements.txt .
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Run app.py when the container launches
CMD ["python", "app/app.py"]