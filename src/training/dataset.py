# from ..preprocessing.data_loader import GCSDataLoader
# from ..preprocessing.image_processor import ImageProcessor
# from ..preprocessing.label_processor import LabelProcessor
# from typing import Tuple, Dict
# import tensorflow as tf
# import numpy as np
# from collections import Counter
# import matplotlib.pyplot as plt


# def create_train_val_dataset(
#     bucket_name: str,
#     num_examples: int,
#     batch_size: int,
#     validation_split: float = 0.2,
#     create_balanced_training_set: bool = False,
#     target_samples_per_class: int = None
# ) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
#     """
#     Creates training and validation datasets from GCS bucket images.

#     Args:
#         bucket_name: GCS bucket name
#         num_examples: Number of examples to load
#         batch_size: Batch size
#         validation_split: Validation split ratio
#         create_balanced_training_set: Whether to balance training dataset
#         target_samples_per_class: Target samples per class if balancing
#     Returns:
#         Train dataset, validation dataset, and number of classes
#     """
#     # Load and process data
#     data_loader = GCSDataLoader(bucket_name)
#     image_processor = ImageProcessor()
#     label_processor = LabelProcessor()

#     images = []
#     labels = []
#     counter = 0
#     while len(images) < num_examples:
#         counter += 1
#         print(f"downloading batch #{counter} size {batch_size}")
#         batch_images, batch_labels = data_loader.load_batch(batch_size)
#         if not batch_images:
#             break

#         processed_images = image_processor.preprocess_batch(
#             tf.convert_to_tensor(batch_images)
#         )
#         processed_labels = [label_processor.process_label(
#             label) for label in batch_labels]

#         images.extend(processed_images.numpy())
#         labels.extend(processed_labels)

#     # Split data
#     val_size = int(len(images) * validation_split)
#     train_images = np.array(images[:-val_size])
#     train_labels = labels[:-val_size]
#     val_images = np.array(images[-val_size:])
#     val_labels = labels[-val_size:]

#     # Create validation dataset
#     val_ds = create_validation_dataset(
#         val_images,
#         val_labels,
#         batch_size
#     )
#     print("\nValidation dataset distribution:")
#     analyze_class_distribution(val_labels)

#     # Create training dataset
#     train_ds = create_training_dataset(
#         train_images,
#         train_labels,
#         batch_size,
#     )
#     print("\nTraining dataset distribution:")
#     analyze_class_distribution(train_labels)

#     return train_ds, val_ds, label_processor.get_num_classes()


# def create_validation_dataset(
#     images: np.ndarray,
#     labels: list,
#     batch_size: int
# ) -> Tuple[tf.data.Dataset, list]:
#     """
#     Creates a validation dataset without augmentation.

#     Args:
#         images: Array of images
#         labels: List of labels
#         batch_size: Batch size for the dataset
#     Returns:
#         Tuple of (TensorFlow dataset, labels list)
#     """
#     val_ds = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(
#         1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     return val_ds


# def create_training_dataset(
#     images: np.ndarray,
#     labels: list,
#     batch_size: int
# ) -> tf.data.Dataset:
#     """Creates a training dataset with augmentation."""
#     augmentation = create_augmentation_layer()

#     # Create dataset with correct order of operations
#     train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
#     train_ds = train_ds.shuffle(1000)
#     train_ds = train_ds.map(
#         lambda x, y: (augmentation(x), y),
#         num_parallel_calls=tf.data.AUTOTUNE
#     )
#     train_ds = train_ds.batch(batch_size)
#     train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

#     return train_ds


# def create_augmentation_layer():
#     """Creates a sequential augmentation layer optimized for dog breed classification"""
#     return tf.keras.Sequential([
#         # Geometric transformations
#         tf.keras.layers.RandomFlip("horizontal"),
#         tf.keras.layers.RandomRotation(0.2),
#         tf.keras.layers.RandomTranslation(0.15, 0.15),
#         tf.keras.layers.RandomZoom(
#             height_factor=(-0.2, -0.1),
#             width_factor=(-0.2, -0.1)
#         ),
#         # Photometric transformations
#         tf.keras.layers.RandomBrightness(0.2),
#         tf.keras.layers.RandomContrast(0.2),
#     ])


# def analyze_class_distribution(labels: list) -> Dict[int, int]:
#     """
#     Analyzes and visualizes the distribution of classes in the dataset.

#     Args:
#         labels: List of labels
#     Returns:
#         Dictionary with class counts
#     """
#     class_distribution = Counter(labels)

#     # Print distribution
#     print("\nClass Distribution:")
#     for class_id, count in class_distribution.items():
#         print(f"Class {class_id}: {count} samples")

#     return class_distribution


# def create_balanced_dataset(images: np.ndarray,
#                             labels: list,
#                             batch_size: int,
#                             target_samples_per_class: int = None) -> Tuple[tf.data.Dataset, list]:
#     """
#     Creates a balanced dataset by augmenting underrepresented classes.

#     Args:
#         images: Array of images
#         labels: List of labels
#         batch_size: Batch size for the dataset
#         target_samples_per_class: Target number of samples per class (if None, uses maximum found)
#     Returns:
#         Tuple of (TensorFlow dataset, balanced labels list)
#     """
#     class_distribution = Counter(labels)

#     if target_samples_per_class is None:
#         target_samples_per_class = max(class_distribution.values())

#     augmentation = tf.keras.Sequential([
#         tf.keras.layers.RandomFlip("horizontal"),
#         tf.keras.layers.RandomRotation(0.2),
#         tf.keras.layers.RandomZoom(0.2),
#         tf.keras.layers.RandomBrightness(0.2),
#         tf.keras.layers.RandomContrast(0.2),
#     ])

#     balanced_images = []
#     balanced_labels = []

#     for class_id in sorted(class_distribution.keys()):
#         class_indices = [i for i, label in enumerate(
#             labels) if label == class_id]
#         class_images = images[class_indices]
#         current_samples = len(class_indices)

#         balanced_images.extend(class_images)
#         balanced_labels.extend([class_id] * current_samples)

#         samples_to_generate = target_samples_per_class - current_samples

#         if samples_to_generate > 0:
#             for _ in range(samples_to_generate):
#                 img_idx = np.random.randint(0, current_samples)
#                 img = class_images[img_idx]
#                 augmented_img = augmentation(tf.expand_dims(img, 0))[0]
#                 balanced_images.append(augmented_img)
#                 balanced_labels.append(class_id)

#     balanced_images = np.array(balanced_images)
#     balanced_labels = np.array(balanced_labels)

#     ds = tf.data.Dataset.from_tensor_slices((balanced_images, balanced_labels))
#     ds = ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

#     return ds, balanced_labels.tolist()

# train.py
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
