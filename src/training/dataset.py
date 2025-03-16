import tensorflow as tf
from typing import Tuple
import json


def parse_tfrecord_fn(example_proto):
    """Parse TFRecord example."""
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    features = tf.io.parse_single_example(example_proto, feature_description)

    # Decode the serialized tensor
    image = tf.io.parse_tensor(features['image'], out_type=tf.uint8)

    # Ensure the shape is correct
    image = tf.ensure_shape(image, [224, 224, 3])

    # Cast to float32 after parsing
    image = tf.cast(image, tf.float32)

    return image, features['label']


def create_training_dataset(
    bucket_name: str,
    tfrecord_path: str,
    labels_path: str,
    batch_size: int,
    validation_split: float = 0.2
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, list]:
    """
    Creates training and validation datasets from a single TFRecord file in GCS.
    """
    # Get full GCS path
    tfrecord_pattern = f"gs://{bucket_name}/{tfrecord_path}/*.tfrecord"

    # Verify TFRecord exists
    if not tf.io.gfile.glob(tfrecord_pattern):
        raise FileNotFoundError(
            f"No TFRecord files found at {tfrecord_pattern}")

    # Create dataset from TFRecords
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(tfrecord_pattern))

    # Count records before parsing
    dataset_size = sum(1 for _ in dataset)
    if dataset_size == 0:
        raise ValueError("TFRecord dataset is empty")

    print(f"Found {dataset_size} records in TFRecord file")

    # Calculate split sizes
    val_size = int(dataset_size * validation_split)

    # Parse TFRecords
    dataset = dataset.map(
        parse_tfrecord_fn,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Split the dataset
    train_ds = dataset.skip(val_size)
    val_ds = dataset.take(val_size)

    # Read class names from metadata file
    label_map_path = f"gs://{bucket_name}/{labels_path}/label_map.json"
    try:
        with tf.io.gfile.GFile(label_map_path, 'r') as f:
            label_map = json.loads(f.read())
            # Convert label_map to ordered list of class names
            class_names = [k for k, v in sorted(
                label_map.items(), key=lambda x: x[1])]
    except Exception as e:
        raise ValueError(
            f"Failed to read label map from {label_map_path}: {str(e)}")

    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # Optimize datasets for training
    train_ds = train_ds.shuffle(1000).batch(
        batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, num_classes, class_names, label_map
