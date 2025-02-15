import tensorflow as tf
import os
import json
from datetime import datetime
from collections import Counter
import numpy as np
from .metrics import TrainingMetrics


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_names: list,
    epochs: int = 20,
) -> tf.keras.Model:
    """Single phase training with metrics tracking"""
    log_dir = f"logs/training/{datetime.now().strftime('%Y%m%d-%H%M')}"
    metrics = TrainingMetrics(log_dir=log_dir)

    # Configure model for transfer learning
    model.trainable = False

    # Train with class weights
    model = train_phase(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=epochs,
        phase_name="Transfer Learning",
        learning_rate=1e-3,
        metrics=metrics,
        class_weight=compute_class_weight(train_ds)
    )

    return model


def train_phase(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    phase_name: str,
    learning_rate: float,
    metrics: TrainingMetrics,
    class_weight: dict = None,
) -> tf.keras.callbacks.History:
    """Training phase with integrated metric tracking"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=metrics.get_metrics()
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=metrics.create_callbacks(),
        class_weight=class_weight
    )

    return model


def create_model(num_classes: int, input_shape: tuple = (224, 224, 3)) -> tf.keras.Model:
    """Creates a MobileNetV3-Small model with preprocessing, augmentation and regularization"""
    inputs = tf.keras.Input(shape=input_shape)

    # Add preprocessing layer
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)

    # Add data augmentation layer (only active during training)
    x = create_augmentation_layer()(x)

    # Create base model
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Base model processing
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Dense layers with regularization
    dense_config = {
        'kernel_regularizer': tf.keras.regularizers.l2(0.001),
        'activation': 'swish'
    }

    x = tf.keras.layers.Dense(256, **dense_config)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(128, **dense_config)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x)

    return tf.keras.Model(inputs, outputs)


def create_augmentation_layer():
    """Creates a sequential augmentation layer optimized for dog breed classification"""
    return tf.keras.Sequential([
        # Geometric transformations
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.05, -0.15),
            width_factor=(-0.05, -0.15)
        ),

        # Photometric transformations
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),

    ])


def compute_class_weight(dataset: tf.data.Dataset) -> dict:
    """
    Compute class weights from a TensorFlow dataset

    Args:
        dataset: TensorFlow dataset containing (image, label) pairs
    Returns:
        Dictionary mapping class indices to weights
    """
    # Extract all labels from dataset
    labels = []
    for _, batch_labels in dataset:
        labels.extend(batch_labels.numpy())

    # Count class frequencies
    counter = Counter(labels)
    total_samples = sum(counter.values())
    n_classes = len(counter)

    # Compute weights inversely proportional to class frequencies
    weights = {
        class_id: total_samples / (n_classes * count)
        for class_id, count in counter.items()
    }

    # print("Computed class weights:", weights)
    return weights


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_names: list,
    epochs: int = 20,
) -> tf.keras.Model:
    """Single phase training with metrics tracking"""
    log_dir = f"logs/training/{datetime.now().strftime('%Y%m%d-%H%M')}"
    metrics = TrainingMetrics(log_dir=log_dir)

    # Configure model for transfer learning
    model.trainable = False

    # Train with class weights
    model = train_phase(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=epochs,
        phase_name="Transfer Learning",
        learning_rate=1e-3,
        metrics=metrics,
        class_weight=compute_class_weight(train_ds)
    )

    return model


def train_phase(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    phase_name: str,
    learning_rate: float,
    metrics: TrainingMetrics,
    class_weight: dict = None,
) -> tf.keras.callbacks.History:
    """Training phase with integrated metric tracking"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=metrics.get_metrics()
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=metrics.create_callbacks(),
        class_weight=class_weight
    )

    return model


def load_or_create_model(num_classes: int,
                         model_gcs_path: str = None) -> tf.keras.Model:
    """
    Loads a previously saved model from GCS if a model_gcs_path is provided.
    Otherwise, creates a new MobileNetV3-based model.

    Args:
        num_classes: Number of output classes.
        input_shape: Input shape of the images.
        model_gcs_path: GCS path to load model from. Expects a 'gs://' URI.
        custom_objects: Dictionary of custom objects needed for loading.

    Returns:
        A tf.keras.Model ready for further fine-tuning.
    """
    if model_gcs_path:
        # Load model from GCS. In VertexAI/Cloud Run GCS URIs can be used, but you can also
        # download the model locally first if necessary.
        print(f"Loading model from {model_gcs_path}")
        model = tf.keras.models.load_model(
            model_gcs_path)
        return model
    else:
        return create_model(num_classes)


def create_augmentation_layer():
    """Creates a sequential augmentation layer optimized for dog breed classification"""
    return tf.keras.Sequential([
        # Geometric transformations
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.05, -0.15),
            width_factor=(-0.05, -0.15)
        ),

        # Photometric transformations
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),

    ])


def compute_class_weight(dataset: tf.data.Dataset) -> dict:
    """
    Compute class weights from a TensorFlow dataset

    Args:
        dataset: TensorFlow dataset containing (image, label) pairs
    Returns:
        Dictionary mapping class indices to weights
    """
    # Extract all labels from dataset
    labels = []
    for _, batch_labels in dataset:
        labels.extend(batch_labels.numpy())

    # Count class frequencies
    counter = Counter(labels)
    total_samples = sum(counter.values())
    n_classes = len(counter)

    # Compute weights inversely proportional to class frequencies
    weights = {
        class_id: total_samples / (n_classes * count)
        for class_id, count in counter.items()
    }

    # print("Computed class weights:", weights)
    return weights


def save_model(model: tf.keras.Model, version: str, bucket_name: str, class_names: list) -> None:
    """
    Saves the trained model in the Keras (.keras) format along with a metadata JSON file.

    Both artifacts are stored in a versioned folder in GCS.

    Args:
        model: The tf.keras.Model to be saved.
        version: Version identifier used as the folder name.
        bucket_name: The target GCS bucket name.
        class_names: List of class names to be saved as metadata.
    """
    # Create a folder path for this version of the model.
    model_dir = f"gs://{bucket_name}/{version}"
    model_path = f"{model_dir}/{version}.keras"

    # Save the model in the native Keras format.
    print(f"Saving model to {model_path}")
    tf.keras.models.save_model(model, model_path, include_optimizer=True)

    # Create metadata and save it as a .json file.
    metadata = {"class_names": class_names}
    metadata_path = f"{model_dir}/metadata.json"
    print(f"Saving metadata to {metadata_path}")

    # Use tf.io.gfile, which supports gs:// URIs, to write the JSON file.
    with tf.io.gfile.GFile(metadata_path, "w") as f:
        json.dump(metadata, f)
