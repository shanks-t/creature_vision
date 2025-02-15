from google.cloud import aiplatform
import tensorflow as tf
from typing import List
from datetime import datetime


class TrainingMetrics:
    def __init__(self, project_id: str, location: str, experiment_name: str):
        # Initialize Vertex AI with default TensorBoard instance
        aiplatform.init(
            project=project_id,
            location=location,
            experiment=experiment_name
        )

        # Get or create experiment
        self.experiment = aiplatform.Experiment.get_or_create(
            display_name=experiment_name,
            description="Transfer learning for image classification"
        )

        # Start a new run
        self.run = self.experiment.start_run(
            run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M')}")

        # Get the backing TensorBoard
        self.tensorboard = self.experiment.get_backing_tensorboard_resource()

    def log_metrics(self, metrics_dict: dict, step: int = None):
        """Log metrics to Vertex AI experiment"""
        self.run.log_metrics(metrics_dict, step=step)

    def get_metrics(self):
        """Essential metrics for multiclass classification"""
        return [
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=5, name='top_5_accuracy'),
            tf.keras.metrics.SparseCategoricalCrossentropy(
                name='cross_entropy')
        ]

    def create_callbacks(self):
        """Create callbacks including custom metric logging"""
        class VertexAICallback(tf.keras.callbacks.Callback):
            def __init__(self, metrics_logger):
                super().__init__()
                self.metrics_logger = metrics_logger

            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    self.metrics_logger.log_metrics(logs, step=epoch)

        return [
            VertexAICallback(self),
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

    def end_run(self):
        """End the current run"""
        self.run.end()
