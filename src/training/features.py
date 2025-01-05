import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds  # This import is missing


def sample_imagenet_dataset(num_samples: int = 1000) -> tf.data.Dataset:
    """
    Create a sampled ImageNet dataset using TensorFlow Datasets

    Args:
        num_samples: Number of samples to take from ImageNet

    Returns:
        TensorFlow dataset containing sampled images
    """
    # Load downsampled ImageNet with proper configuration
    imagenet = tfds.load(
        'imagenet2012',  # Specify a valid subset
        split='train',
        shuffle_files=True,
        as_supervised=True  # Returns (image, label) pairs
    )

    # Match MobileNetV3's expected preprocessing
    imagenet = imagenet.map(
        # Match MobileNetV3 input size
        lambda x, y: (tf.image.resize(x, (224, 224)), y)
    )

    # Take random sample with proper shuffling
    sampled_ds = imagenet.shuffle(
        buffer_size=10000,
        reshuffle_each_iteration=True
    ).take(num_samples)

    # Batch and prefetch for better performance
    return sampled_ds.batch(32).prefetch(tf.data.AUTOTUNE)


def extract_features(model: tf.keras.Model, dataset: tf.data.Dataset, num_samples: int = 300) -> np.ndarray:
    """Extract features using MobileNetV3's penultimate layer"""
    # Get the layer before classification head
    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer('global_average_pooling2d').output
    )

    # Add ImageNet preprocessing
    preprocess = tf.keras.applications.mobilenet_v3.preprocess_input

    features = []
    for images, _ in dataset.take(num_samples).batch(32):
        # Apply proper preprocessing chain
        images = tf.image.resize(images, (224, 224))  # Ensure consistent size
        processed_images = preprocess(images)
        batch_features = feature_model.predict(processed_images, verbose=0)
        features.append(batch_features)

    return np.concatenate(features, axis=0)


def domain_distance(
    source_dataset: tf.data.Dataset,
    target_dataset: tf.data.Dataset,
    base_model: tf.keras.Model,  # Only need one model
    num_samples: int = 1000
) -> float:
    """Compute asymmetric domain distance D(T|S) as defined in paper"""
    source_features = extract_features(base_model, source_dataset, num_samples)
    target_features = extract_features(base_model, target_dataset, num_samples)

    # Compute distances from each target sample to closest source sample
    distances = []
    for t in target_features:
        # Compute euclidean distances to all source samples
        dists = np.linalg.norm(source_features - t, axis=1)
        # Get distance to closest source sample
        min_dist = np.min(dists)
        distances.append(min_dist)

    # Return average minimum distance
    return np.mean(distances)


def evaluate_domain_similarity(
    source_dataset: tf.data.Dataset,
    target_dataset: tf.data.Dataset,
    base_model: tf.keras.Model,
) -> dict:
    """
    Evaluate domain similarity and provide metrics

    Args:
        source_dataset: Source domain dataset
        target_dataset: Target domain dataset
        base_model: Base model for feature extraction

    Returns:
        Dictionary containing domain similarity metrics
    """
    # Compute bidirectional domain distances
    source_to_target = domain_distance(
        source_dataset, target_dataset, base_model)
    target_to_source = domain_distance(
        target_dataset, source_dataset, base_model)

    # Compute symmetric distance
    symmetric_distance = (source_to_target + target_to_source) / 2

    return {
        'source_to_target_distance': float(source_to_target),
        'target_to_source_distance': float(target_to_source),
        'symmetric_distance': float(symmetric_distance)
    }
