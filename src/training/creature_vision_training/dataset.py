import json
from typing import Any

import tensorflow as tf


def parse_tfrecord_fn(example_proto):
    """Parse TFRecord example."""
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    features = tf.io.parse_single_example(example_proto, feature_description)

    # Decode the serialized tensor
    image = tf.io.parse_tensor(features["image"], out_type=tf.uint8)

    # Ensure the shape is correct
    image = tf.ensure_shape(image, [224, 224, 3])

    # Cast to float32 after parsing
    # image = tf.cast(image, tf.float32)

    return image, features["label"]


def get_augmentation_fn():
    """Returns a function that applies data augmentation."""
    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.05, -0.15), width_factor=(-0.05, -0.15)
            ),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomContrast(0.2),
        ]
    )

    def augment_fn(image, label):
        # Ensures augmentation is applied
        image = augment(image, training=True)
        return image, label

    return augment_fn


def create_training_dataset(
    bucket_name: str,
    tfrecord_path: str,
    labels_path: str,
    batch_size: int,
    validation_split: float = 0.2,
    random_seed: int = 42,
    gpu: bool = False,
) -> tuple[tf.data.Dataset, tf.data.Dataset, Any]:
    """
    Creates training and validation datasets from a processed TFRecord directory.
    """
    # Get all TFRecord files
    tfrecords = tf.io.gfile.glob(f"gs://{bucket_name}/{tfrecord_path}/*.tfrecord")

    if not tfrecords:
        raise FileNotFoundError(f"No TFRecord files found at {tfrecord_path}")

    dataset = tf.data.Dataset.from_tensor_slices(tfrecords)
    dataset = dataset.interleave(
        lambda path: tf.data.TFRecordDataset(path).prefetch(1),
        cycle_length=min(len(tfrecords), 8),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    metadata_path = f"gs://{bucket_name}/{tfrecord_path}/metadata.json"
    try:
        with tf.io.gfile.GFile(metadata_path) as f:
            dataset_size = int(json.load(f)["dataset_size"])
    except Exception:
        # ponytail: fallback scan for old processed folders without metadata.json.
        dataset_size = sum(1 for _ in dataset)
        dataset = tf.data.Dataset.from_tensor_slices(tfrecords).interleave(
            lambda path: tf.data.TFRecordDataset(path).prefetch(1),
            cycle_length=min(len(tfrecords), 8),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

    if dataset_size == 0:
        raise ValueError("Combined TFRecord dataset is empty")

    prefetch_buffer = tf.data.AUTOTUNE if gpu else 1

    # Parse and split dataset
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(
        buffer_size=dataset_size,
        seed=random_seed,
        reshuffle_each_iteration=False,
    )
    val_size = int(dataset_size * validation_split)
    if dataset_size > 1:
        val_size = min(max(val_size, 1), dataset_size - 1)

    train_ds = dataset.skip(val_size)
    val_ds = dataset.take(val_size)

    # Label map loading
    label_map_path = f"gs://{bucket_name}/{labels_path}/label_map.json"
    try:
        with tf.io.gfile.GFile(label_map_path, "r") as f:
            label_map = json.loads(f.read())
    except Exception as e:
        raise ValueError(f"Failed to read label map from {label_map_path}") from e

    # Augmentation and batching
    augment_fn = get_augmentation_fn()
    train_ds = train_ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = (
        train_ds.shuffle(buffer_size=dataset_size)
        .batch(batch_size)
        .prefetch(prefetch_buffer)
    )
    val_ds = val_ds.batch(batch_size).prefetch(prefetch_buffer)

    return train_ds, val_ds, label_map
