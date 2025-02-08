from google.cloud import storage
import tensorflow as tf
import json
import os


def load_model(
    version: str,
    bucket_name: str = "tf_models_cv"
) -> tuple[tf.keras.Model, dict]:
    """
    Loads model and metadata from GCS bucket.

    Args:
        version: Model version to load
        bucket_name: GCS bucket name
    Returns:
        Tuple of (loaded model, metadata dict)
    """
    model_dir = f"./models/{version}"
    os.makedirs(model_dir, exist_ok=True)

    # Download from GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Download model
    model_blob = bucket.blob(f"{version}/model.keras")
    model_path = f"{model_dir}/model.keras"
    model_blob.download_to_filename(model_path)

    # Download metadata
    metadata_blob = bucket.blob(f"{version}/metadata.json")
    metadata_path = f"{model_dir}/metadata.json"
    metadata_blob.download_to_filename(metadata_path)

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Load model with custom objects
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'preprocess_input': tf.keras.applications.mobilenet_v3.preprocess_input
        }
    )

    return model, metadata
