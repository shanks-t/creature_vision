# preprocessing/data_loader.py
import numpy as np
from google.cloud import storage
from PIL import Image
import json
import io
import random
from datetime import datetime
from .logging_utils import setup_logger, timer, log_execution_time
import concurrent.futures

logger = setup_logger('GCSDataLoader')


class GCSDataLoader:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.client = storage.Client(project="creature-vision")
        self.bucket = self.client.bucket(self.bucket_name)
        self.logger = logger
        self._image_paths_cache = None
        self._last_cache_update = None

    def _update_image_paths_cache(self):
        """Cache image paths with pagination to avoid memory issues"""
        with timer(self.logger, 'Updating image paths cache'):
            image_paths = []

            # Use prefix iterator to handle pagination efficiently
            for prefix in ['correct_predictions/', 'incorrect_predictions/']:
                blobs = self.bucket.list_blobs(
                    prefix=prefix,
                    fields='items(name)',  # Only fetch the name field
                    page_size=1000  # Adjust based on your needs
                )

                # Filter during iteration to avoid loading all blobs into memory
                for blob in blobs:
                    if blob.name.endswith('.jpg'):
                        image_paths.append(blob.name)

            self._image_paths_cache = image_paths
            self._last_cache_update = datetime.now()
            self.logger.info(f"Cache updated with {len(image_paths)} images")

    def _get_random_batch_paths(self, batch_size: int) -> list:
        """Get random batch of image paths from cache"""
        if (self._image_paths_cache is None or
                (datetime.now() - self._last_cache_update).seconds > 3600):  # Cache for 1 hour
            self._update_image_paths_cache()

        return random.sample(self._image_paths_cache, batch_size)

    @log_execution_time(logger)
    def _load_raw_batch(self, batch_size: int):
        """Load a batch of raw images and labels using parallel downloads"""
        image_paths = self._get_random_batch_paths(batch_size)

        # Prepare concurrent futures for parallel downloads
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(batch_size, 10)) as executor:
            # Submit all image and label downloads
            future_to_path = {
                executor.submit(self._load_raw_sample, path, path.replace('.jpg', '_labels.json')): path
                for path in image_paths
            }

            images = []
            labels = []

            # Process completed futures as they come in
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    image, label = future.result()
                    images.append(image)
                    labels.append(label)
                    self.logger.info(
                        f"Successfully loaded image {len(images)}/{batch_size}")
                except Exception as e:
                    self.logger.error(f"Error loading {path}: {str(e)}")

        self.logger.info(f"Loaded batch of {len(images)} images")
        return images, labels

    @log_execution_time(logger)
    def _load_raw_sample(self, image_path: str, label_path: str):
        """Load a single image and its corresponding label concurrently"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both downloads concurrently
            image_future = executor.submit(self._download_image, image_path)
            label_future = executor.submit(self._download_label, label_path)

            # Wait for both to complete
            image_array = image_future.result()
            label = label_future.result()

        return image_array, label

    def _download_image(self, image_path: str) -> np.ndarray:
        """Download and process single image"""
        with timer(self.logger, f'Loading image {image_path}'):
            image_blob = self.bucket.blob(image_path)
            image_bytes = image_blob.download_as_bytes()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return np.array(image)

    def _download_label(self, label_path: str) -> dict:
        """Download and process single label"""
        with timer(self.logger, f'Loading label {label_path}'):
            label_blob = self.bucket.blob(label_path)
            label_json = label_blob.download_as_text()
            return json.loads(label_json)
