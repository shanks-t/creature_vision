import json
import logging
import os

import tensorflow as tf
from google.cloud import storage


def load_model(
    version: str, bucket_name: str | None = None
) -> tuple[tf.keras.Model, dict]:
    """
    Loads model and metadata from a GCS bucket.

    Args:
        version: Model version to load (also serves as the folder name in GCS).
        bucket_name: Name of the GCS bucket.

    Returns:
        Tuple of (loaded model, metadata dict).
    """
    # Create a local directory for this model version.
    model_dir = "./model"
    os.makedirs(model_dir, exist_ok=True)

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "creature-vision")
    bucket_name = bucket_name or os.environ.get("MODEL_BUCKET", "tf_models_cv")
    label_bucket_name = os.environ.get(
        "DATA_BUCKET_NAME", "creature-vision-training-set"
    )

    # Set up the GCS client.
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    label_bucket = storage_client.bucket(label_bucket_name)

    # Define local paths for the model and metadata.
    model_file = f"{version}.keras"
    metadata_file = "label_map.json"
    model_path = os.path.join(model_dir, model_file)
    metadata_path = os.path.join(model_dir, metadata_file)

    try:
        # Download the model file from GCS.
        model_blob = bucket.blob(f"{version}/{model_file}")
        logging.info(
            "Downloading model from gs://%s/%s/%s to %s.",
            bucket_name,
            version,
            model_file,
            model_path,
        )
        model_blob.download_to_filename(model_path)
    except Exception as e:
        logging.error(f"Failed to download model: {e}")
        raise

    try:
        # Download the metadata file from GCS.
        metadata_blob = label_bucket.blob(f"processed/metadata/{metadata_file}")
        logging.info(
            "Downloading metadata from gs://%s/processed/metadata/%s to %s.",
            label_bucket_name,
            metadata_file,
            metadata_path,
        )
        metadata_blob.download_to_filename(metadata_path)
    except Exception as e:
        logging.error(f"Failed to download metadata: {e}")
        raise

    # Load metadata from the JSON file.
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except Exception as e:
        logging.error(f"Error loading metadata from {metadata_path}: {e}")
        raise

    # Load the model, including any custom objects required.
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "preprocess_input": tf.keras.applications.mobilenet_v3.preprocess_input
            },
        )
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        raise

    logging.info("Model and metadata loaded successfully.")
    return model, metadata
