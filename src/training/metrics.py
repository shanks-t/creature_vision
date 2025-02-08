"""Simplified F1 score tracking for TensorBoard integration"""
import tensorflow as tf
from typing import List


class TrainingMetrics:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def get_metrics(self):
        return [
            'accuracy',
            'sparse_top_k_categorical_accuracy',
            tf.keras.metrics.SparseCategoricalAccuracy(
                name='categorical_accuracy')
        ]

    def create_callbacks(self):
        return [
            tf.keras.callbacks.TensorBoard(
                log_dir=self.log_dir,
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
