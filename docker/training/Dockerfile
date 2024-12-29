FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ .
ENTRYPOINT ["python", "training/main.py"]
