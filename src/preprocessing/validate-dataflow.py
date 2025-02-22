import tensorflow as tf
from google.cloud import storage
import numpy as np
import os


class TFRecordValidator:
    def __init__(self, bucket_name: str, tfrecord_path: str):
        """Initialize validator with GCS bucket and tfrecord path"""
        self.bucket_name = bucket_name
        self.tfrecord_path = tfrecord_path
        self.full_path = f"gs://{bucket_name}/{tfrecord_path}"

    def _parse_tfrecord(self, example_proto):
        """Parse the input tf.Example proto."""
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        try:
            parsed_features = tf.io.parse_single_example(
                example_proto, feature_description)

            # Decode the image
            image = tf.io.parse_tensor(
                parsed_features['image'], out_type=tf.uint8)
            image = tf.cast(image, tf.float32)
            label = parsed_features['label']
            return image, label
        except tf.errors.InvalidArgumentError as e:
            print(f"Error parsing example: {e}")
            return None, None

    def check_file_exists(self):
        """Verify the TFRecord file exists and print its size"""
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.get_blob(self.tfrecord_path)

        if blob.exists():
            print(f"Found TFRecord file. Size: {blob.size / 1024:.2f} KB")
            return True
        else:
            print(f"TFRecord file not found at {self.full_path}")
            return False

    def validate_tfrecords(self, num_samples: int = 5):
        """Validate TFRecords by loading and inspecting samples"""
        if not self.check_file_exists():
            return

        print(f"\nAttempting to read from: {self.full_path}")

        try:
            # Create TFRecord dataset with error handling
            dataset = tf.data.TFRecordDataset([self.full_path])

            # Count total records first
            total_records = sum(1 for _ in dataset)
            print(f"\nTotal records found: {total_records}")

            if total_records == 0:
                print("WARNING: TFRecord file exists but contains no records")
                return

            print(
                f"\nValidating {min(num_samples, total_records)} samples from TFRecords...")

            # Reset dataset for sample inspection
            dataset = tf.data.TFRecordDataset([self.full_path])
            parsed_dataset = dataset.map(self._parse_tfrecord)

            for i, (image, label) in enumerate(parsed_dataset.take(num_samples)):
                if image is None:
                    continue

                print(f"\nSample {i+1}:")
                print(f"Image shape: {image.shape}")
                print(f"Image dtype: {image.dtype}")
                print(
                    f"Image value range: [{tf.reduce_min(image).numpy():.2f}, {tf.reduce_max(image).numpy():.2f}]")
                print(f"Label: {label.numpy()}")

                # Basic image validation
                if tf.reduce_any(tf.math.is_nan(image)):
                    print("WARNING: Image contains NaN values")
                if tf.reduce_any(tf.math.is_inf(image)):
                    print("WARNING: Image contains Inf values")

        except tf.errors.OpError as e:
            print(f"TensorFlow error reading TFRecord: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


# Usage
if __name__ == "__main__":
    validator = TFRecordValidator(
        bucket_name="creature-vision-training-set",
        tfrecord_path="processed/weekly_20250222/data-00000-of-00001.tfrecord"
    )
    validator.validate_tfrecords()
