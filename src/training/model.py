import tensorflow as tf
import os
from google.cloud import storage
from google.cloud import bigquery
from datetime import datetime


def create_model(num_classes: int, input_shape: tuple = (224, 224, 3)) -> tf.keras.Model:
    """Creates a MobileNetV3-Small model with regularization and normalization"""
    # Create augmentation layer
    data_augmentation = create_augmentation_layer()

    # Create base model
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Create new model with regularization and batch normalization
    inputs = tf.keras.Input(shape=input_shape)

    # Apply augmentation only during training
    x = data_augmentation(inputs)

    # Normalize pixel values
    x = tf.keras.layers.Rescaling(1./255)(x)

    # Base model
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Dense layers with L2 regularization and batch normalization
    x = tf.keras.layers.Dense(
        256,
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(
        128,
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x)

    return tf.keras.Model(inputs, outputs), base_model


def train_model_progressively(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    initial_epochs: int = 10,
    fine_tuning_epochs: int = 5
) -> tf.keras.callbacks.History:
    """Progressive training in two phases"""
    # Phase 1: Train only top layers
    base_model.trainable = False
    initial_history = train_phase(
        model, train_ds, val_ds,
        initial_epochs, "Phase 1: Training top layers",
        learning_rate=1e-4
    )

    # Phase 2: Fine-tune last few layers
    base_model.trainable = True
    # Freeze all layers except the last 4
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    fine_tune_history = train_phase(
        model, train_ds, val_ds,
        fine_tuning_epochs, "Phase 2: Fine-tuning",
        learning_rate=1e-5
    )

    return initial_history, fine_tune_history


def train_phase(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    phase_name: str,
    learning_rate: float
) -> tf.keras.callbacks.History:
    """Train a single phase with regularization"""
    print(f"\nStarting {phase_name}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy',
                 'sparse_top_k_categorical_accuracy'],
        # Add regularization losses automatically
        run_eagerly=True
    )

    callbacks = [
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

    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )


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


def save_model_training_metrics(
    model_version: str,
    accuracy: float,
    top_k_accuracy: float,
) -> None:
    """
    Saves model training metrics to BigQuery.

    Args:
        accuracy: Final model accuracy
        top_k_accuracy: Final top-k accuracy
        model_version: Version string of the model
    """
    client = bigquery.Client()
    table_id = "dog_prediction_app.training_metrics"

    rows_to_insert = [{
        "model_version": model_version,
        "accuracy": accuracy,
        "top_k_accuracy": top_k_accuracy,
        "timestamp": datetime.now().isoformat()
    }]
    print(f"saving training data to bq: {rows_to_insert}")
    errors = client.insert_rows_json(table_id, rows_to_insert)
    if errors:
        raise RuntimeError(f"Error inserting rows to BigQuery: {errors}")


def create_augmentation_layer():
    """Creates a sequential augmentation layer"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.05, -0.15),
            width_factor=(-0.05, -0.15)
        ),
        tf.keras.layers.RandomTranslation(0.1, 0.1)
    ])
