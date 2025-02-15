from google.cloud import aiplatform
import tensorflow as tf
from typing import List


class TrainingMetrics:
    def __init__(self, project_id: str, location: str, experiment_name: str):
        # Initialize Vertex AI
        aiplatform.init(
            project=project_id,
            location=location
        )

        # Create or get existing experiment
        self.experiment = aiplatform.Experiment(experiment_name)

        # Create TensorBoard instance
        self.tensorboard = aiplatform.Tensorboard.create(
            display_name=experiment_name,
            project=project_id,
            location=location
        )

    def get_metrics(self):
        return [
            'accuracy',
            'sparse_top_k_categorical_accuracy',
            tf.keras.metrics.SparseCategoricalAccuracy(
                name='categorical_accuracy')
        ]

    def create_callbacks(self):
        return [
            # Use Vertex AI TensorBoard callback
            tf.keras.callbacks.TensorBoard(
                log_dir=self.tensorboard.log_dir,
                histogram_freq=1,
                update_freq='epoch'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2
            )
        ]

    def log_metric(self, name: str, value: float, step: int = None):
        """Log custom metrics during training"""
        with self.experiment.log_run("training") as run:
            run.log_metric(name, value, step=step)
