import argparse
import datetime
import os
import sys

from google.cloud import aiplatform

from creature_vision_training.model import (
    setup_model,
    run_training,
    load_or_create_model,
    save_model,
    compute_class_weight,
)
from creature_vision_training.dataset import create_training_dataset

# Add GCS module path
sys.path.append("/gcs/creture-vision-ml-artifacts/src/training")


def parse_args():
    """Parse command-line arguments for training."""

    print(f"Raw sys.argv received: {sys.argv}")
    parser = argparse.ArgumentParser(
        description="Train a machine learning model and save it to GCS."
    )
    parser.add_argument(
        "--version", type=str, required=True, help="Version identifier for the model"
    )
    parser.add_argument(
        "--previous_model_version",
        type=str,
        default=None,
        required=False,
        help="Previous model version for stateful training",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Environment configuration
    BUCKET_NAME = os.getenv("TRAINING_BUCKET", "creature-vision-training-set")
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "creature-vision")
    LOCATION = os.getenv("CLOUD_LOCATION", "us-central1")
    MODEL_BUCKET = os.getenv("MODEL_BUCKET", "tf_models_cv")
    STAGING_BUCKET = os.getenv("STAGING_BUCKET", "creture-vision-ml-artifacts")
    AIP_TENSORBOARD_LOG_DIR = os.getenv(
        "AIP_TENSORBOARD_LOG_DIR", "gs://creture-vision-ml-artifacts/local"
    )
    NEW_VERSION = args.version
    if args.previous_model_version == "None":
        args.previous_model_version = None

    print(f"Starting training with version: {args.version}")
    print(f"Previous model version: {args.previous_model_version}")

    experiment_config = {
        "experiment_name": f"{NEW_VERSION.replace('_', '-')}",
        "project_id": PROJECT_ID,
        "location": LOCATION,
        "staging_bucket": STAGING_BUCKET,
    }

    # Dataset preparation
    train_ds, val_ds, label_map = create_training_dataset(
        bucket_name=BUCKET_NAME,
        tfrecord_path="processed",
        labels_path="processed/metadata",
        batch_size=BATCH_SIZE,
        model_version=NEW_VERSION,
        validation_split=0.3,
    )

    # Model initialization
    model = load_or_create_model(label_map, prev_version=args.previous_model_version)
    # Initialize Vertex AI experiment
    aiplatform.init(
        experiment=experiment_config["experiment_name"],
        project=experiment_config["project_id"],
        location=experiment_config["location"],
        staging_bucket=experiment_config["staging_bucket"],
    )

    # Training execution
    with aiplatform.start_run(
        f"training-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}".lower()
    ) as run:
        # Model and experiment setup
        model, metrics, callbacks = setup_model(model)

        # Class weighting
        class_weights = compute_class_weight(train_ds, label_map)

        # Run training phase
        trained_model = run_training(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            metrics=metrics,
            callbacks=callbacks,
            class_weight=class_weights,
            epochs=100,
            learning_rate=1e-3,
        )

        # Log key parameters
        run.log_params(
            {
                "batch_size": BATCH_SIZE,
                "base_model": "MobileNetV3Small",
                "trainable_layers": 3,
                "class_weight_strategy": "inverse_frequency",
            }
        )

    # Model persistence
    save_model(
        model=trained_model,
        version=NEW_VERSION,
        bucket_name=MODEL_BUCKET,
    )


if __name__ == "__main__":
    main()
