import tensorflow as tf
from google.cloud import bigquery
from datetime import datetime


def save_training_metrics_and_cm(
    confusion_matrix: tf.Tensor,
    model_version: str,
    accuracy: float,
    top_k_accuracy: float,
    label_names: list = None
) -> None:
    """
    Saves training metrics and confusion matrix to BigQuery.

    Args:
        confusion_matrix: TensorFlow confusion matrix
        model_version: Version string of the model
        accuracy: Final model accuracy
        top_k_accuracy: Final top-k accuracy
        label_names: List of class names corresponding to matrix indices
    """
    client = bigquery.Client()
    table_id = "dog_prediction_app.training_metrics"

    # Convert confusion matrix to list format
    cm_data = confusion_matrix.numpy()

    # Create confusion matrix string in BigQuery ML format
    cm_rows = []
    for true_idx, row in enumerate(cm_data):
        for pred_idx, count in enumerate(row):
            if count > 0:  # Only store non-zero entries
                true_label = label_names[true_idx] if label_names else str(
                    true_idx)
                pred_label = label_names[pred_idx] if label_names else str(
                    pred_idx)
                cm_rows.append(f"{true_label}:{pred_label}:{count}")

    confusion_matrix_str = "|".join(cm_rows)

    # Combine metrics and confusion matrix in single row
    row = {
        "model_version": model_version,
        "accuracy": accuracy,
        "top_k_accuracy": top_k_accuracy,
        "confusion_matrix": confusion_matrix_str,
        "timestamp": datetime.now().isoformat()
    }

    errors = client.insert_rows_json(table_id, [row])
    if errors:
        raise RuntimeError(f"Error inserting metrics to BigQuery: {errors}")
