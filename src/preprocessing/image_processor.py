# preprocessing/image_processor.py
import numpy as np
from PIL import Image
import tensorflow as tf


class ImageProcessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    # preprocessing/image_processor.py


class ImageProcessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def preprocess_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Preprocess a batch of images for MobileNetV3.
        """
        if images.ndim != 4:
            raise ValueError("Expected 4D array for batch processing")

        # Convert to float32
        images = tf.cast(images, tf.float32)

        return images
