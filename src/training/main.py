from .model import train_model_progressively, create_model, save_model, save_training_metrics_and_cm, evaluate_base_model
from .dataset import create_training_dataset
from .features import evaluate_domain_similarity, sample_imagenet_dataset
import numpy as np


def main():
    BUCKET_NAME = "creature-vision-training-set"
    NUM_EXAMPLES = 320
    BATCH_SIZE = 32

    # # Create dataset
    train_ds, val_ds, num_classes, class_names = create_training_dataset(
        BUCKET_NAME,
        NUM_EXAMPLES,
        BATCH_SIZE
    )

    # for images, labels in train_ds.take(1):
    #     print(f"labels: {labels}")
    #     print(f"Batch shape: {images.shape}")
    #     print(f"Labels shape: {labels.shape}")
    #     print(f"Number of classes: {num_classes}")
    # for images, labels in val_ds.take(1):
    #     print(f"Batch shape: {images.shape}")
    #     print(f"Labels shape: {labels.shape}")
    #     print(f"Number of classes: {num_classes}")

    # Create and train model
    model, base_model = create_model(num_classes)
    baseline_metrics = evaluate_base_model(val_ds, class_names)
    print("Baseline metrics:", baseline_metrics)
    print(f"TensorBoard logs available at: {baseline_metrics['log_dir']}")

    inital_history, fine_tuned_history, final_cm = train_model_progressively(
        model, base_model, train_ds, val_ds, class_names)
    # model_version = "m_net_v3.1_test"
    # save_model(model, model_version)
    # save_training_metrics_and_cm(model_version=model_version, confusion_matrix=final_cm,
    #                              accuracy=fine_tuned_history.history['sparse_categorical_accuracy'][-1], top_k_accuracy=fine_tuned_history.history['sparse_top_k_categorical_accuracy'][-1])

    # # Print training results
    # print("Final accuracy:",
    #       inital_history.history['sparse_categorical_accuracy'][-1])
    # print("Final top-k accuracy:",
    #       inital_history.history['sparse_top_k_categorical_accuracy'][-1])
    # print("Final accuracy:",
    #       fine_tuned_history.history['sparse_categorical_accuracy'][-1])
    # print("Final top-k accuracy:",
    #       fine_tuned_history.history['sparse_top_k_categorical_accuracy'][-1])


if __name__ == "__main__":
    main()
