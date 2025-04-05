import os
import json
import tensorflow as tf
from google.cloud import storage
import logging


def load_model(
    version: str, bucket_name: str = "tf_models_cv"
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

    # Set up the GCS client.
    storage_client = storage.Client(project="creature-vision")
    bucket = storage_client.bucket(bucket_name)
    label_bucket = storage_client.bucket("creature-vision-training-set")

    # Define local paths for the model and metadata.
    model_file = f"{version}.keras"
    metadata_file = "label_map.json"
    model_path = os.path.join(model_dir, model_file)
    metadata_path = os.path.join(model_dir, metadata_file)

    try:
        # Download the model file from GCS.
        model_blob = bucket.blob(f"{version}/{model_file}")
        logging.info(
            f"Downloading model from gs://{bucket_name}/{version}/{model_file} to {model_path}."
        )
        model_blob.download_to_filename(model_path)
    except Exception as e:
        logging.error(f"Failed to download model: {e}")
        raise

    try:
        # Download the metadata file from GCS.
        metadata_blob = label_bucket.blob(f"processed/metadata/{metadata_file}")
        logging.info(
            f"Downloading metadata from gs://{label_bucket}processed/metadata/{metadata_file} to {metadata_path}."
        )
        metadata_blob.download_to_filename(metadata_path)
    except Exception as e:
        logging.error(f"Failed to download metadata: {e}")
        raise

    # Load metadata from the JSON file.
    try:
        with open(metadata_path, "r") as f:
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
