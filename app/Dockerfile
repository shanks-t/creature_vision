# Use the official lightweight Python image.
FROM python:3.10-slim

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

WORKDIR /app

COPY requirements.txt .
RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY app.py ./app/

# Run app.py when the container launches
CMD ["python", "app/app.py"]