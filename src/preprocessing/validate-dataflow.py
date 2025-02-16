import tensorflow as tf
from google.cloud import storage
import matplotlib.pyplot as plt
import numpy as np


def validate_tfrecords(bucket_name: str, num_samples: int = 5):
    """Validate TFRecords by inspecting samples and metadata"""
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix="processed_test/")
    file_patterns = [
        f"gs://{bucket_name}/{b.name}" for b in blobs if b.name.endswith('.tfrecord')]

    # Parse function matching your TFRecord schema
    def parse_tfrecord(example):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        parsed = tf.io.parse_single_example(example, feature_description)
        image = tf.io.decode_jpeg(parsed['image'])  # Match your encoding
        return image, parsed['label']

    # Create validation dataset
    dataset = tf.data.TFRecordDataset(file_patterns)
    dataset = dataset.map(parse_tfrecord)

    # Basic validation checks
    print(f"Found {len(file_patterns)} TFRecord files")
    element_count = sum(1 for _ in dataset)
    print(
        f"Total examples in TFRecords: {element_count} (should match your test limit)")

    # Verify label distribution
    label_counts = {}
    label_examples = {}
    for image, label in dataset:
        label_id = int(label.numpy())
        if label_id not in label_counts:
            label_counts[label_id] = 0
            label_examples[label_id] = image.numpy()
        label_counts[label_id] += 1

    print("\nLabel Distribution:")
    for label_id, count in label_counts.items():
        print(f"Label {label_id}: {count} examples")

    # Visual inspection of samples
    for idx, (image, label) in enumerate(dataset.take(num_samples)):
        print(f"\nSample {idx+1}:")
        print(f"Label: {label.numpy()}")
        print(f"Image shape: {image.shape}")
        print(f"Image dtype: {image.dtype}")
        print(f"Pixel range: {np.min(image)}-{np.max(image)}")

        # Save sample images locally
        plt.imshow(image.numpy().astype("uint8"))
        plt.savefig(f"sample_{idx}.png")
        plt.close()


if __name__ == "__main__":
    validate_tfrecords("creature-vision-training-set", num_samples=3)
