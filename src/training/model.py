import tensorflow as tf
import os
import io
from google.cloud import storage
from datetime import datetime
from collections import Counter
from src.training.monitor import *
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.metrics import F1Score


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
        'kernel_regularizer': tf.keras.regularizers.l2(0.01),
        'activation': 'swish'  # Simplified activation approach
    }

    x = tf.keras.layers.Dense(256, **dense_config)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(128, **dense_config)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x)

    return tf.keras.Model(inputs, outputs), base_model


def evaluate_base_model(val_ds: tf.data.Dataset,
                        class_names: list) -> dict:
    """
    Evaluate the base MobileNetV3-Small on validation dataset
    """
    num_classes = len(class_names)

    # Create complete model with original classification head
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    # eval model with new classification head
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    # Set up TensorBoard callback
    log_dir = "logs/baseline/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    )

    # Compile with metrics
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'sparse_categorical_accuracy']
    )

    # Evaluate
    results = model.evaluate(val_ds, callbacks=[tensorboard_callback])

    # Get predictions for a batch of validation data
    for images, labels in val_ds.take(1):
        predictions = model.predict(images)
        predicted_classes = tf.argmax(predictions, axis=-1)

        # Format the prediction data
        actuals = [class_names[label] for label in labels.numpy()]
        predicted = [class_names[pred] for pred in predicted_classes.numpy()]
        visualization_data = format_prediction_data(
            actuals,
            predicted,
            len(labels)
        )

        # Log predictions vs actuals to TensorBoard
        with tf.summary.create_file_writer(log_dir + "/predictions").as_default():
            tf.summary.text(
                "Predictions_vs_Actuals",
                tf.convert_to_tensor(str(visualization_data)),
                step=0
            )

    return {
        'loss': results[0],
        'initial_accuracy': results[1],
        'initial_top_k_accuracy': results[2],
        'log_dir': log_dir
    }


def train_model_progressively(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_names: list,
    initial_epochs: int = 10,
    fine_tuning_epochs: int = 10,
) -> tuple:
    """Progressive training with visualization support"""

    # Set up TensorBoard logging
    log_dir = f"logs/training/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_images=False
    )

    # Phase 1: Train only top layers
    base_model.trainable = False
    initial_history, initial_cm = train_phase(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=initial_epochs,
        phase_name="Phase 1: Training top layers",
        learning_rate=1e-3,
        class_names=class_names,
        tensorboard_callback=tensorboard_callback
    )

    # Phase 2: Fine-tuning with class weights
    base_model.trainable = True
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    class_weight = compute_class_weight(train_ds)

    fine_tune_history, fine_tune_cm = train_phase(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=fine_tuning_epochs,
        phase_name="Phase 2: Fine-tuning",
        learning_rate=1e-4,
        class_weight=class_weight,
        class_names=class_names,
        tensorboard_callback=tensorboard_callback
    )

    return initial_history, fine_tune_history, fine_tune_cm


def train_phase(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    phase_name: str,
    learning_rate: float,
    class_names: list,
    tensorboard_callback: tf.keras.callbacks.TensorBoard,
    class_weight: dict = None,
) -> tuple:
    """Train a single phase with visualization support"""

    callbacks = [
        tensorboard_callback,
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

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight
    )

    # Get predictions for a batch of validation data
    for images, labels in val_ds.take(1):
        predictions = model.predict(images)
        predicted_classes = tf.argmax(predictions, axis=-1)

        # Format prediction data
        actuals = [class_names[label] for label in labels.numpy()]
        predicted = [class_names[pred] for pred in predicted_classes.numpy()]
        visualization_data = format_prediction_data(
            actuals,
            predicted,
            len(labels)
        )

        # Log predictions vs actuals to TensorBoard
        with tf.summary.create_file_writer(tensorboard_callback.log_dir).as_default():
            tf.summary.text(
                f"{phase_name}/predictions_vs_actuals",
                tf.convert_to_tensor(visualization_data),
                step=epochs
            )

        # In train_phase:
        cm = compute_confusion_matrix(model, val_ds)
        cm_image = plot_confusion_matrix(cm, class_names)

        with tf.summary.create_file_writer(tensorboard_callback.log_dir).as_default():
            # Log class names
            tf.summary.text(
                f"{phase_name}/class_names",
                tf.convert_to_tensor(str(class_names)),
                step=0
            )
            # Log confusion matrix as image
            tf.summary.image(
                f"{phase_name}/confusion_matrix",
                cm_image,
                step=epochs
            )

    return history, cm


def save_model(
    model: tf.keras.Model,
    version: str,
    bucket_name: str = "tf_models_cv"
) -> None:
    """
    Saves model locally and uploads to GCS bucket.

    Args:
        model: Trained Keras model
        version: Model version string
        bucket_name: GCS bucket name for model storage
    """
    # Save locally
    print(f"saving model locally..")
    local_path = f"./models/{version}.keras"
    os.makedirs("models", exist_ok=True)
    model.save(local_path)

    # Upload to GCS
    print(f"saving model to gcs..")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{version}.keras")
    blob.upload_from_filename(local_path)


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


def compute_confusion_matrix(model, dataset):
    """Compute confusion matrix from a dataset."""
    true_labels = []
    predictions = []

    for images, labels in dataset:
        preds = model.predict(images)
        pred_labels = tf.argmax(preds, axis=1)

        true_labels.extend(labels.numpy())
        predictions.extend(pred_labels.numpy())

    # Create confusion matrix
    cm = tf.math.confusion_matrix(true_labels, predictions)

    return cm


def format_prediction_data(actuals, predictions, batch_size):
    """Format prediction data for clearer TensorBoard visualization"""
    formatted_data = []
    for i in range(batch_size):
        formatted_data.append(
            f"Image {i+1}:\n  Actual: {actuals[i]}\n  Predicted: {predictions[i]}\n")
    return "\n".join(formatted_data)


def plot_confusion_matrix(cm, class_names):
    """Convert confusion matrix to a TensorBoard-compatible image"""
    # Create a new figure
    cm = cm.numpy()
    plt.figure(figsize=(10, 10))

    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    # Set up the axes
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    # Add labels
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    # Convert plot to PNG image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image
