import tensorflow as tf
import os
import json
from collections import Counter
from google.cloud import aiplatform


def setup_model(
    model: tf.keras.Model,
) -> tuple[tf.keras.Model, list, list]:
    """Configures model and Vertex AI experiment without starting run"""

    # Define metrics
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=5, name='top_5_accuracy'),
        tf.keras.metrics.SparseCategoricalCrossentropy(
            name='cross_entropy')
    ]

    # Create callbacks
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=os.environ['AIP_TENSORBOARD_LOG_DIR'],
            histogram_freq=1,
            profile_batch=(50, 100),
            update_freq='epoch'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            min_delta=0.001
        )
    ]

    # Freeze base layers
    for layer in model.layers[:-3]:
        layer.trainable = False

    return model, metrics, callbacks


def run_training(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    metrics: list,
    callbacks: list,
    class_weight: dict,
    epochs: int = 20,
    learning_rate: float = 1e-3
) -> tf.keras.Model:
    """Executes training within Vertex AI run context"""
    # Compile and train inside run context
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=metrics
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
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


def load_or_create_model(num_classes: int, model_gcs_path: str = None) -> tf.keras.Model:
    """Loads existing model or creates new one with dynamic class adaptation"""
    if model_gcs_path:
        try:
            # Load base model with preserved architecture
            base_model = tf.keras.models.load_model(model_gcs_path)
            print(f"Loaded base model from {model_gcs_path}")

            # Extract penultimate layer outputs
            penultimate_output = base_model.layers[-2].output

            # Create new classifier head
            new_output = tf.keras.layers.Dense(
                num_classes,
                activation='softmax',
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
                name='dynamic_classifier'
            )(penultimate_output)

            # Reconstruct full model
            model = tf.keras.Model(
                inputs=base_model.input,
                outputs=new_output,
                name=base_model.name + "_adapted"
            )

            # Preserve previous layer weights
            for layer in model.layers[:-1]:
                layer.set_weights(base_model.get_layer(
                    layer.name).get_weights())

        except Exception as e:
            print(f"Error adapting model: {str(e)}")
    else:
        model = create_model(num_classes)

    return model


def create_augmentation_layer():
    """Creates a sequential augmentation layer optimized for image classification"""
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
