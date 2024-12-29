from .model import train_model_progressively, create_model, save_model, save_model_training_metrics
from .dataset import create_training_dataset


def main():
    BUCKET_NAME = "creature-vision-training-set"
    NUM_EXAMPLES = 3000
    BATCH_SIZE = 32

    # Create dataset
    train_ds, val_ds, num_classes = create_training_dataset(
        BUCKET_NAME,
        NUM_EXAMPLES,
        BATCH_SIZE
    )

    for images, labels in train_ds.take(1):
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Number of classes: {num_classes}")
    for images, labels in val_ds.take(1):
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Number of classes: {num_classes}")

    # Create and train model
    model, base_model = create_model(num_classes)
    inital_history, fine_tuned_history = train_model_progressively(
        model, base_model, train_ds, val_ds)
    model_version = "m_net_v3.1"
    # save_model(model, model_version)
    # save_model_training_metrics(model_version,
    # history.history['sparse_categorical_accuracy'][-1], history.history['sparse_top_k_categorical_accuracy'][-1])

    # Print training results
    print("Final accuracy:",
          inital_history.history['sparse_categorical_accuracy'][-1])
    print("Final top-k accuracy:",
          inital_history.history['sparse_top_k_categorical_accuracy'][-1])
    print("Final accuracy:",
          fine_tuned_history.history['sparse_categorical_accuracy'][-1])
    print("Final top-k accuracy:",
          fine_tuned_history.history['sparse_top_k_categorical_accuracy'][-1])


if __name__ == "__main__":
    main()
