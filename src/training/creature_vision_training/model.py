import re
import os
from collections import Counter
from io import StringIO

import tensorflow as tf
from google.cloud import aiplatform


def setup_model(
    model: tf.keras.Model,
    base_learning_rate: float = 1e-3,
) -> tuple[tf.keras.Model, list, list]:
    """Configures model and Vertex AI experiment without starting run"""

    # Define metrics
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy"),
        tf.keras.metrics.SparseCategoricalCrossentropy(name="cross_entropy"),
    ]
    log_dir = os.getenv(
        "AIP_TENSORBOARD_LOG_DIR", "gs://creture-vision-ml-artifacts/local"
    )

    # Learning rate schedule (optional)
    def lr_schedule(epoch, lr):
        if epoch > 10:
            return lr * 0.5
        return lr

    # Callbacks
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            profile_batch=(50, 100),
            update_freq="epoch",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, min_delta=0.001, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
        tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1),
    ]

    return model, metrics, callbacks


def run_training(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    metrics: list,
    callbacks: list,
    class_weight: dict,
    epochs: int = 100,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Executes training within Vertex AI run context"""
    # Compile and train inside run context
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=metrics,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )

    return model


def create_model(
    num_classes: int, input_shape: tuple = (224, 224, 3)
) -> tf.keras.Model:
    """Creates a MobileNetV3-Small model with preprocessing, augmentation and regularization"""
    inputs = tf.keras.Input(shape=input_shape)

    # Add preprocessing layer
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)

    # Create base model
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )

    # Base model processing
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Dense layers with regularization
    dense_config = {
        "kernel_regularizer": tf.keras.regularizers.l2(0.001),
        "activation": "swish",
    }

    x = tf.keras.layers.Dense(256, **dense_config)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(128, **dense_config)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
    )(x)

    return tf.keras.Model(inputs, outputs)


def parse_model_version(version_str: str) -> tuple:
    """Parses version string like 'v3_1' -> (3, 1)"""
    match = re.match(r"v(\d+)_(\d+)", version_str)
    if not match:
        raise ValueError(f"Invalid model version format: {version_str}")
    return int(match.group(1)), int(match.group(2))


def load_or_create_model(label_map: dict, prev_version: str) -> tf.keras.Model:
    """
    Loads an existing model or creates a new one with conditional classifier logic:
    - If version == 'v3_0': recreate MobileNetV3 backbone from scratch and attach new classifier
    - If version >= 'v3_1': load full model from GCS and optionally unfreeze last N layers
    """
    num_classes = len(label_map)
    print(f"number of classes: {num_classes}")

    # Fallback: start from scratch if no prior version
    if not prev_version or prev_version == "None":
        print("No previous version provided — creating a new base model.")
        return create_model(num_classes)

    try:
        # Parse version string
        major, minor = parse_model_version(prev_version)

        if (major, minor) == (3, 0):
            print("v3_0 detected — creating a new base model.")
            # Directly call `create_model` instead of manually loading or building the model
            model = create_model(num_classes)
            print("Model created with MobileNetV3 backbone and custom classifier.")

        else:
            print("Stateful retrain (v3_1 or higher) — loading full model.")
            model_gcs_path = f"gs://tf_models_cv/{prev_version}/{prev_version}.keras"
            model = tf.keras.models.load_model(model_gcs_path)
            print(f"Loaded model from {model_gcs_path}")

            # Optionally: unfreeze last N layers for fine-tuning
            unfreeze_count = 20
            for layer in model.layers[-unfreeze_count:]:
                layer.trainable = True
            print(f"Unfroze last {unfreeze_count} layers for fine-tuning.")

    except Exception as e:
        print(f"Error loading or adapting model: {e}")
        print("Falling back to creating a new model.")
        model = create_model(num_classes)

    return model


def compute_class_weight(dataset, label_map):
    """
    Compute class weights ensuring all class keys are accounted for.

    Args:
        dataset: TensorFlow dataset containing (image, label) pairs.
        label_map: Dictionary mapping class names to integer labels.

    Returns:
        Dictionary mapping class indices to weights.
    """
    # Extract class indices from label_map
    all_classes = set(label_map.values())  # Set of all possible classes

    # Extract all labels from dataset
    labels = []
    for _, batch_labels in dataset:
        labels.extend(batch_labels.numpy())

    # Convert all labels to integers (to match label_map values)
    labels = [int(label) for label in labels]

    # Count occurrences of each label in the dataset
    counter = Counter(labels)
    total_samples = sum(counter.values())

    # Compute weights inversely proportional to class frequency
    weights = {
        class_id: total_samples
        / (len(all_classes) * counter.get(class_id, 1))  # Default 1 if missing
        for class_id in all_classes
    }

    # Normalize weights to prevent extreme values
    max_weight = max(weights.values())
    weights = {k: v / max_weight for k, v in weights.items()}  # Normalize
    # print(f"class weights {weights}")

    return weights


def save_model(
    model: tf.keras.Model,
    version: str,
    bucket_name: str,
) -> None:
    """
    Saves the trained model in the Keras (.keras) format and its model summary as a .txt file.

    Artifacts are stored in a versioned folder in GCS.

    Args:
        model: The tf.keras.Model to be saved.
        version: Version identifier used as the folder name.
        bucket_name: The target GCS bucket name.
    """
    # GCS paths
    model_dir = f"gs://{bucket_name}/{version}"
    model_path = f"{model_dir}/{version}.keras"
    summary_path = f"{model_dir}/{version}_summary.txt"

    # === Save Model ===
    print(f"Saving model to {model_path}")
    tf.keras.models.save_model(model, model_path, include_optimizer=True)

    # === Capture Model Summary ===
    stream = StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_text = stream.getvalue()
    stream.close()

    print(f"Saving model summary to {summary_path}")
    with tf.io.gfile.GFile(summary_path, "w") as f:
        f.write(summary_text)
