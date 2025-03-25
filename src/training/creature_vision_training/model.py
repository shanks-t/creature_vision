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
    log_dir = os.getenv('AIP_TENSORBOARD_LOG_DIR',
                        'gs://creture-vision-ml-artifacts/local')

    # Create callbacks
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
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


def load_or_create_model(num_classes: int, prev_version) -> tf.keras.Model:
    """Loads existing model or creates new one with dynamic class adaptation"""

    # Be defensive: treat "None" string or empty string as no version
    if prev_version and prev_version != "None":
        try:
            model_gcs_path = f"gs://tf_models_cv/{prev_version}"
            base_model = tf.keras.models.load_model(model_gcs_path)
            print(f"Loaded base model from {model_gcs_path}")

            penultimate_output = base_model.layers[-2].output

            new_output = tf.keras.layers.Dense(
                num_classes,
                activation='softmax',
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
                name='dynamic_classifier'
            )(penultimate_output)

            model = tf.keras.Model(
                inputs=base_model.input,
                outputs=new_output,
                name=base_model.name + "_adapted"
            )

            for layer in model.layers[:-1]:
                layer.set_weights(base_model.get_layer(
                    layer.name).get_weights())

        except Exception as e:
            print(f"Error adapting model: {str(e)}")
            print(f"Falling back to creating a new base model...")
            model = create_model(num_classes)

    else:
        print("No previous model version provided. Creating new base model...")
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
        class_id: total_samples /
        (len(all_classes) * counter.get(class_id, 1))  # Default 1 if missing
        for class_id in all_classes
    }

    # Normalize weights to prevent extreme values
    max_weight = max(weights.values())
    weights = {k: v / max_weight for k, v in weights.items()}  # Normalize
    print(f"class weights {weights}")

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
