import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO

import numpy as np
import requests
import structlog
import tensorflow as tf
from flask import Flask, jsonify
from fuzzywuzzy import fuzz
from google.cloud import bigquery, storage
from PIL import Image

try:
    from inference.load_model import load_model
except ImportError:  # Docker runs this module from src/inference.
    from load_model import load_model

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

@dataclass(frozen=True)
class Settings:
    project_id: str
    model_version: str
    data_bucket_name: str = "creature-vision-training-set"
    metrics_dataset: str = "dog_prediction_app"
    metrics_table: str = "prediction_metrics"


def load_settings() -> Settings:
    model_version = os.environ.get("MODEL_VERSION")
    if not model_version:
        raise ValueError("MODEL_VERSION environment variable not set")

    return Settings(
        project_id=os.environ.get("GOOGLE_CLOUD_PROJECT", "creature-vision"),
        model_version=model_version,
        data_bucket_name=os.environ.get(
            "DATA_BUCKET_NAME", "creature-vision-training-set"
        ),
        metrics_dataset=os.environ.get("BQ_METRICS_DATASET", "dog_prediction_app"),
        metrics_table=os.environ.get("BQ_METRICS_TABLE", "prediction_metrics"),
    )


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
def insert_prediction_data(
    bigquery_client,
    settings: Settings,
    actual,
    predicted,
    is_correct,
    latency,
    confidence,
):
    table_ref = bigquery_client.dataset(settings.metrics_dataset).table(
        settings.metrics_table
    )
    table = bigquery_client.get_table(table_ref)

    rows_to_insert = [
        {
            "timestamp": datetime.now(),
            "actual": actual,
            "predicted": predicted,
            "is_correct": is_correct,
            "latency": latency,
            "confidence": confidence,
            "model_version": settings.model_version,
        }
    ]

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
        if data["status"] == "success":
            image_url = data["message"]
            breed = image_url.split("/")[4].replace("-", " ")
        else:
            raise ValueError("Failed to fetch dog image from the API")

        img_response = requests.get(image_url)
        img_response.raise_for_status()
        img = Image.open(BytesIO(img_response.content))
        img = img.convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)

        return img_array, img, breed

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch dog image: {str(e)}") from e


# Function to save images and labels in GCS
def save_image_and_label(data_bucket, img, model_label, api_label, is_correct, version):
    """Store the prediction results in GCS."""
    directory = (
        f"{version}/correct_predictions"
        if is_correct
        else f"{version}/incorrect_predictions"
    )
    timestamp = int(time.time())
    base_filename = (
        f"{model_label}_{timestamp}" if is_correct else f"{api_label}_{timestamp}"
    )

    # Save image to GCS
    img_blob_name = f"{directory}/{base_filename}.jpg"
    img_blob = data_bucket.blob(img_blob_name)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()
    img_blob.upload_from_string(img_byte_arr, content_type="image/jpeg")

    # Save labels to GCS
    label_blob_name = f"{directory}/{base_filename}_labels.json"
    label_blob = data_bucket.blob(label_blob_name)
    label_data = {"model_label": model_label, "api_label": api_label}
    label_blob.upload_from_string(
        json.dumps(label_data), content_type="application/json"
    )


LABEL_MAPPINGS = {
    "Saint_Bernard": "stbernard",
    "german shepherd": "germanshepherd",
    "alaskan malamute": "malamute",
    "West_Highland_white_terrier": "terrier westhighland",
    "blldog boston": "Boston_bull",
    "terrier scottish": "Scotch_terrier",
    "hound blood": "bloodhound",
    "terrier kerryblue": "Kerry_blue_terrier",
}


def normalize_breed_name(breed):
    """Normalize breed labels and return as a set of lowercase words."""
    label = breed.lower().replace("_", " ").replace("-", " ").strip()
    normalized = LABEL_MAPPINGS.get(label, label)
    return set(normalized.split())


def fuzzy_match(actual, predicted, threshold=85):
    print("first string comparison false. Running fuzzy match...")
    ratio = fuzz.ratio(actual.lower(), predicted.lower())
    print(f"Fuzzy match ratio: {ratio}")
    return ratio >= threshold


# Function to compare actual and predicted breeds
def compare_breeds(actual, predicted, version):
    """Check if the predicted breed matches the actual breed."""
    if version == "v3_0":
        actual_set = normalize_breed_name(actual)
        predicted_set = normalize_breed_name(predicted)
        strict_match_threshold = 0.75
        # Check for exact match
        if actual_set == predicted_set:
            return True

        # Check for strict partial match
        common_words = actual_set.intersection(predicted_set)

        # Handle empty sets after exclusion
        if not actual_set or not predicted_set:
            return actual.lower() == predicted.lower()

        actual_coverage = len(common_words) / len(actual_set)
        predicted_coverage = len(common_words) / len(predicted_set)

        if (
            actual_coverage >= strict_match_threshold
            and predicted_coverage >= strict_match_threshold
        ):
            return True

        # Handle cases where one set is a subset of the other
        if actual_set.issubset(predicted_set) or predicted_set.issubset(actual_set):
            return True

        return fuzzy_match(actual, predicted)
    else:
        return actual == predicted


# Function to make predictions
def predict_breed(
    model: tf.keras.Model, img_array: np.ndarray, metadata: dict, version: str
) -> tuple[str, float]:
    """
    Run inference using either base MobileNetV3-Small or a custom model.

    Args:
        model (tf.keras.Model): Trained or pretrained model for inference.
        img_array (np.ndarray): Preprocessed input image array.
        metadata (dict): Dictionary containing 'class_names' for custom models.
        version (str): String indicating the model version, 'v3_0' for base MobileNetV3.

    Returns:
        tuple[str, float]: Predicted breed/class label and confidence score.
    """
    predictions = model.predict(img_array)
    predictions = predictions[0]

    predicted_idx = np.argmax(predictions)
    confidence = float(predictions[predicted_idx])

    if version == "v3_0":
        # Built-in decode_predictions for base MobileNetV3/ImageNet
        decoded_predictions = tf.keras.applications.imagenet_utils.decode_predictions(
            np.expand_dims(predictions, axis=0), top=1
        )[0][0]
        predicted_label = decoded_predictions[1]  # Human-readable label
    elif metadata and isinstance(metadata, dict):
        # Assume label-to-index format
        label_map = {v: k for k, v in metadata.items()}  # reverse index to label
        predicted_label = label_map.get(predicted_idx)
    else:
        raise ValueError(
            "Invalid configuration: For custom models, provide metadata['class_names']."
        )

    return predicted_label, confidence


def create_app(
    settings: Settings | None = None,
    model: tf.keras.Model | None = None,
    metadata: dict | None = None,
    bigquery_client=None,
    storage_client=None,
) -> Flask:
    settings = settings or load_settings()
    bigquery_client = bigquery_client or bigquery.Client(project=settings.project_id)
    storage_client = storage_client or storage.Client(project=settings.project_id)
    data_bucket = storage_client.bucket(settings.data_bucket_name)

    if model is None:
        model, metadata = initialize_model(settings.model_version)

    app = Flask(__name__)

    @app.route("/predict/", methods=["GET"])
    @app.route("/predict", methods=["GET"])
    def run_prediction():
        """Run inference and return results."""
        try:
            start_time = time.time()
            img_array, img, api_label = load_random_dog_image()

            model_label, confidence = predict_breed(
                model, img_array, metadata, settings.model_version
            )
            is_correct = compare_breeds(api_label, model_label, settings.model_version)
            latency = time.time() - start_time

            result = {
                "model_version": settings.model_version,
                "is_correct": is_correct,
                "actual": api_label,
                "predicted": model_label,
                "confidence": confidence,
                "latency": latency,
            }

            if settings.model_version != "v3_0":
                save_image_and_label(
                    data_bucket,
                    img,
                    model_label,
                    api_label,
                    is_correct,
                    settings.model_version,
                )
            insert_prediction_data(
                bigquery_client,
                settings,
                api_label,
                model_label,
                is_correct,
                latency,
                confidence,
            )

            logger.info(
                "Prediction made",
                model_version=settings.model_version,
                actual=api_label,
                predicted=model_label,
                is_correct=is_correct,
                confidence=confidence,
                latency=latency,
            )

            return jsonify(result)

        except Exception as e:
            logger.exception("Error during prediction", error=str(e))
            return jsonify({"error": str(e)}), 500

    @app.route("/", methods=["GET"])
    def health_check():
        return "Service is running", 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
    )
