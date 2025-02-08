from .model import train_model, create_model, save_model
from .dataset import create_training_dataset
import datetime


def main():
    BUCKET_NAME = "creature-vision-training-set"
    NUM_EXAMPLES = 32
    BATCH_SIZE = 8
    VERSION = f"v1_{datetime.datetime.now().strftime('%Y%m%d')}"

    # Create dataset
    train_ds, val_ds, num_classes, class_names = create_training_dataset(
        BUCKET_NAME,
        NUM_EXAMPLES,
        BATCH_SIZE
    )

    # Create and train model
    model = create_model(num_classes)

    train_model(
        model, train_ds, val_ds, class_names
    )

    # save fine-tuned model to gcs
    save_model(model, class_names, version=VERSION, bucket_name="tf_models_cv")


if __name__ == "__main__":
    main()
