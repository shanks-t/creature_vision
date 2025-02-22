from .model import setup_model, run_training, load_or_create_model, save_model, compute_class_weight
from .dataset import create_training_dataset
from google.cloud import aiplatform
import datetime
import os


def main():
    # Environment configuration
    BUCKET_NAME = os.getenv("TRAINING_BUCKET", "creature-vision-training-set")
    NUM_EXAMPLES = int(os.getenv("NUM_EXAMPLES", "32"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "creature-vision")
    LOCATION = os.getenv("CLOUD_LOCATION", "us-central1")
    MODEL_BUCKET = os.getenv("MODEL_BUCKET", "tf_models_cv")
    STAGING_BUCKET = os.getenv("STAGING_BUCKET", "creture-vision-ml-artifacts")
    model_gcs_path = "gs://tf_models_cv/feb-22-2025/feb-22-2025.keras"
    NEW_VERSION = f"{datetime.datetime.now().strftime('%b-%d-%Y')}".lower()
    experiment_config = {
        "experiment_name": f"{NEW_VERSION}",
        "project_id": PROJECT_ID,
        "location": LOCATION,
        "staging_bucket": STAGING_BUCKET
    }

    # Dataset preparation
    train_ds, val_ds, num_classes, class_names = create_training_dataset(
        bucket_name=BUCKET_NAME,
        tfrecord_path="processed/weekly_20250222",
        labels_path="processed/metadata",
        batch_size=BATCH_SIZE,
        validation_split=0.2
    )

    # Model initialization
    model = load_or_create_model(num_classes, model_gcs_path=model_gcs_path)
    # Initialize Vertex AI experiment
    aiplatform.init(
        experiment=experiment_config["experiment_name"],
        project=experiment_config["project_id"],
        location=experiment_config["location"],
        staging_bucket=experiment_config["staging_bucket"]
    )

    # Training execution
    with aiplatform.start_run(f"training-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}".lower()) as run:
        # Model and experiment setup
        model, metrics, callbacks = setup_model(model)

        # Class weighting
        class_weights = compute_class_weight(train_ds)

        # Run training phase
        trained_model = run_training(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            metrics=metrics,
            callbacks=callbacks,
            class_weight=class_weights,
            epochs=20,
            learning_rate=1e-3
        )

        # Log key parameters
        run.log_params({
            "batch_size": BATCH_SIZE,
            "base_model": "MobileNetV3Small",
            "trainable_layers": 3,
            "class_weight_strategy": "inverse_frequency"
        })

    # Model persistence
    save_model(
        model=trained_model,
        version=NEW_VERSION,
        bucket_name=MODEL_BUCKET,
        class_names=class_names
    )


if __name__ == "__main__":
    main()
