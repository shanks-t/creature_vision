import argparse
import requests
import time
import json
import structlog
from datetime import datetime
import os

import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from google.cloud import storage
from flask import Flask, jsonify
from google.cloud import bigquery


from load_model import load_model

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Initialize Flask app
app = Flask(__name__)

# Initialize Google Cloud clients
bigquery_client = bigquery.Client(project="creature-vision")
storage_client = storage.Client(project="creature-vision")

# Define storage bucket
DATA_BUCKET_NAME = "creature-vision-training-set"
data_bucket = storage_client.bucket(DATA_BUCKET_NAME)


# Load model and metadata
def initialize_model(model_version):
    """Load the trained model from GCS based on the provided version."""
    try:
        model, metadata = load_model(model_version)
        logger.info("Model loaded successfully", version=model_version)
        return model, metadata
    except Exception as e:
        logger.error("Failed to load model", error=str(e))
        raise

# Function to insert prediction data into BigQuery


def insert_prediction_data(actual, predicted, is_correct, latency, confidence, model_version):
    dataset_id = "dog_prediction_app"
    table_id = "prediction_metrics"
    table_ref = bigquery_client.dataset(dataset_id).table(table_id)
    table = bigquery_client.get_table(table_ref)

    rows_to_insert = [{
        "timestamp": datetime.now(),
        "actual": actual,
        "predicted": predicted,
        "is_correct": is_correct,
        "latency": latency,
        "confidence": confidence,
        "model_version": model_version
    }]

    errors = bigquery_client.insert_rows(table, rows_to_insert)
    if errors:
        logger.error(f"Encountered errors while inserting rows: {errors}")

# Function to load a random dog image from an API


def load_random_dog_image():
    """Fetch a random dog image and process it for inference."""
    dog_api_url = "https://dog.ceo/api/breeds/image/random"

    try:
        response = requests.get(dog_api_url)
        response.raise_for_status()
        data = response.json()
        if data['status'] == 'success':
            image_url = data['message']
            breed = image_url.split('/')[4].replace('-', ' ')
        else:
            raise ValueError("Failed to fetch dog image from the API")

        img_response = requests.get(image_url)
        img_response.raise_for_status()
        img = Image.open(BytesIO(img_response.content))
        img = img.convert('RGB')
        img = img.resize((224, 224))

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        img_array = tf.keras.applications.mobilenet_v3.preprocess_input(
            img_array)

        return img_array, img, breed

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch dog image: {str(e)}")

# Function to save images and labels in GCS


def save_image_and_label(img, model_label, api_label, is_correct):
    """Store the prediction results in GCS."""
    directory = "correct_predictions" if is_correct else "incorrect_predictions"
    timestamp = int(time.time())
    base_filename = f"{model_label}_{timestamp}" if is_correct else f"{api_label}_{timestamp}"

    # Save image to GCS
    img_blob_name = f"{directory}/{base_filename}.jpg"
    img_blob = data_bucket.blob(img_blob_name)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    img_blob.upload_from_string(img_byte_arr, content_type='image/jpeg')

    # Save labels to GCS
    label_blob_name = f"{directory}/{base_filename}_labels.json"
    label_blob = data_bucket.blob(label_blob_name)
    label_data = {
        "model_label": model_label,
        "api_label": api_label
    }
    label_blob.upload_from_string(json.dumps(
        label_data), content_type='application/json')

# Function to compare actual and predicted breeds


def compare_breeds(actual, predicted):
    """Check if the predicted breed matches the actual breed."""
    return actual == predicted

# Function to make predictions


def predict_breed(model: tf.keras.Model, img_array: np.ndarray, metadata: dict) -> tuple[str, float]:
    """Run inference using the trained model."""
    predictions = model.predict(img_array)
    predicted_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_idx])
    predicted_breed = metadata['class_names'][predicted_idx]
    return predicted_breed, confidence

# Define API endpoint for predictions


@app.route('/predict/', methods=['GET'])
@app.route('/predict', methods=['GET'])
def run_prediction():
    """Run inference and return results."""
    try:
        start_time = time.time()
        img_array, img, api_label = load_random_dog_image()

        model_label, confidence = predict_breed(model, img_array, metadata)
        is_correct = compare_breeds(api_label, model_label)
        latency = time.time() - start_time

        result = {
            "model_version": model_version,
            "is_correct": is_correct,
            "actual": api_label,
            "predicted": model_label,
            "confidence": confidence,
            "latency": latency
        }

        save_image_and_label(img, model_label, api_label, is_correct)
        insert_prediction_data(api_label, model_label,
                               is_correct, latency, confidence, model_version)

        logger.info("Prediction made",
                    model_version=model_version,
                    actual=api_label,
                    predicted=model_label,
                    is_correct=is_correct,
                    confidence=confidence,
                    latency=latency)

        return jsonify(result)

    except Exception as e:
        logger.exception("Error during prediction", error=str(e))
        return jsonify({"error": str(e)}), 500

# Health check endpoint


@app.route('/', methods=['GET'])
def health_check():
    return "Service is running", 200


if __name__ == "__main__":
    # use env vars
    model_version = os.environ.get("MODEL_VERSION")
    if not model_version:
        raise ValueError("MODEL_VERSION environment variable not set")

    # Load model
    model, metadata = initialize_model(model_version)

    # Start Flask server
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
