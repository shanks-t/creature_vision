# preprocessing/data_loader.py
import numpy as np
from google.cloud import storage
from PIL import Image
import json
import io
import random


class GCSDataLoader:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.client = storage.Client(project="creature-vision")
        self.bucket = self.client.bucket(self.bucket_name)

    def _load_raw_sample(self, image_path: str, label_path: str):
        """Load a single image and its corresponding label"""
        # Load image
        image_blob = self.bucket.blob(image_path)
        image_bytes = image_blob.download_as_bytes()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_array = np.array(image)

        # Load label
        label_blob = self.bucket.blob(label_path)
        label_json = label_blob.download_as_text()
        label = json.loads(label_json)

        return image_array, label

    def _load_raw_batch(self, batch_size: int):
        """Load a batch of raw images and labels"""
        blobs = list(self.bucket.list_blobs(prefix='correct_predictions/'))
        blobs.extend(list(self.bucket.list_blobs(
            prefix='incorrect_predictions/')))

        image_blobs = [blob for blob in blobs if blob.name.endswith('.jpg')]
        random.shuffle(image_blobs)

        images = []
        labels = []

        for image_blob in image_blobs[:batch_size]:
            image_path = image_blob.name
            label_path = image_path.replace('.jpg', '_labels.json')

            try:
                image, label = self._load_raw_sample(image_path, label_path)
                images.append(image)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {image_path}: {str(e)}")

            if len(images) == batch_size:
                break

        return images, labels

    def load_single_sample(self, image_path: str, label_path: str):
        """Load a single image and its corresponding label"""
        return self._load_raw_sample(image_path, label_path)

    def load_batch(self, batch_size: int):
        """Load a batch of images and labels"""
        return self._load_raw_batch(batch_size)
