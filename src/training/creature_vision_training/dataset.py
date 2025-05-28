from typing import Tuple, Any
import json

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
        images = augment(image, training=True)
        return images, label

    return augment_fn


def create_training_dataset(
    bucket_name: str,
    tfrecord_path: str,
    labels_path: str,
    batch_size: int,
    validation_split: float = 0.2,
    gpu: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Any]:
    """
    Creates training and validation datasets that includes all TFRecords for the
    given model_version and a percentage of TFRecords from other versions.
    """

    # Get all TFRecord files
    tfrecords = tf.io.gfile.glob(f"gs://{bucket_name}/{tfrecord_path}/*.tfrecord")
    print(f"tf record path: {tfrecord_path}")
    if not tfrecords:
        raise FileNotFoundError(f"No TFRecord files found at {tfrecord_path}")

    dataset = tf.data.Dataset.from_tensor_slices(tfrecords)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x).prefetch(1),
        cycle_length=min(len(tfrecords), 8),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    with tf.io.gfile.GFile(f"gs://{bucket_name}/{tfrecord_path}/metadata.json") as f:
        meta = json.load(f)
        dataset_size = meta["dataset_size"]

    # Set autotune parameters for gpu
    # num_parallel_calls = tf.data.AUTOTUNE if gpu else 4
    # shuffle_buffer = 2048 if gpu else 1000
    # prefetch_buffer = tf.data.AUTOTUNE if gpu else 1
    # batch_size_opt = 64 if gpu else batch_size

    # Parse and split dataset
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    val_size = int(dataset_size * validation_split)

    train_ds = dataset.skip(val_size)
    val_ds = dataset.take(val_size)

    steps_per_epoch = (dataset_size - val_size) // batch_size
    validation_steps = val_size // batch_size

    # Label map loading
    label_map_path = f"gs://{bucket_name}/{labels_path}/label_map.json"
    try:
        with tf.io.gfile.GFile(label_map_path, "r") as f:
            label_map = json.loads(f.read())
    except Exception as e:
        raise ValueError(f"Failed to read label map from {label_map_path}: {str(e)}")

    # Augmentation and batching
    augment_fn = get_augmentation_fn()
    train_ds = (
        train_ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=1000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, label_map
