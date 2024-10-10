# Use the official lightweight Python image.
FROM python:3.10-slim

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

WORKDIR /app

COPY mobile_net_v3_small.keras ./mobile_net_v3_small.keras

COPY ./app/requirements.txt .
RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY ./app/main.py ./app/

# Run app.py when the container launches
CMD ["python", "app/main.py"]