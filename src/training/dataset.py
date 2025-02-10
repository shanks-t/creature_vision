import tensorflow as tf
from ..preprocessing.data_loader import GCSDataLoader
from ..preprocessing.image_processor import ImageProcessor
from ..preprocessing.label_processor import LabelProcessor
from typing import Tuple


def create_training_dataset(
    bucket_name: str,
    num_examples: int,
    batch_size: int,
    validation_split: float = 0.2
) -> Tuple[tf.data.Dataset, int]:
    """
    Creates a training dataset from GCS bucket images.
    Returns the dataset and number of classes.
    """

    # Initialize processors
    data_loader = GCSDataLoader(bucket_name)
    image_processor = ImageProcessor()
    label_processor = LabelProcessor()

    # Load and process data
    images = []
    labels = []
    counter = 0
    while len(images) < num_examples:
        counter += 1
        print(f"downloading batch #{counter} size {batch_size}")
        batch_images, batch_labels = data_loader.load_batch(batch_size)
        if not batch_images:
            break

        # Process images
        processed_images = image_processor.preprocess_batch(
            tf.convert_to_tensor(batch_images)
        )

        # Process labels
        processed_labels = [
            label_processor.process_label(label)
            for label in batch_labels
        ]

        images.extend(processed_images.numpy())
        labels.extend(processed_labels)

        # Calculate split
    val_size = int(len(images) * validation_split)

    # Split data
    train_images = images[:-val_size]
    train_labels = labels[:-val_size]
    val_images = images[-val_size:]
    val_labels = labels[-val_size:]

    class_names = label_processor.get_class_names()
    # print("class names:", class_names)

    # # Create augmentation layer
    # data_augmentation = create_augmentation_layer()

    # Create TF datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    # Apply augmentation only to training data
    # train_ds = train_ds.map(
    #     lambda x, y: (data_augmentation(tf.expand_dims(x, 0))[0], y),
    #     num_parallel_calls=tf.data.AUTOTUNE
    # )
    train_ds = train_ds.shuffle(1000).batch(
        batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_ds = val_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, label_processor.get_num_classes(), class_names


# def create_augmentation_layer():
#     """Creates a sequential augmentation layer optimized for dog breed classification"""
#     return tf.keras.Sequential([
#         tf.keras.layers.RandomFlip("horizontal"),
#         tf.keras.layers.RandomRotation(0.2),
#         tf.keras.layers.RandomTranslation(0.15, 0.15),
#         tf.keras.layers.RandomZoom(
#             height_factor=(-0.2, -0.1),
#             width_factor=(-0.2, -0.1)
#         ),
#         tf.keras.layers.RandomBrightness(0.2),
#         tf.keras.layers.RandomContrast(0.2),
#     ])
