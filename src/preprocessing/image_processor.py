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

    # ... (rest of the class remains the same)

    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to a single image.

        Args:
            image: numpy array of shape (height, width, 3)

        Returns:
            Augmented image as numpy array
        """
        # Convert to tensor
        img_tensor = tf.convert_to_tensor(image)

        # Random flip left/right
        if tf.random.uniform(()) > 0.5:
            img_tensor = tf.image.flip_left_right(img_tensor)

        # Random brightness
        img_tensor = tf.image.random_brightness(img_tensor, 0.2)

        # Random contrast
        img_tensor = tf.image.random_contrast(img_tensor, 0.8, 1.2)

        # Random saturation
        img_tensor = tf.image.random_saturation(img_tensor, 0.8, 1.2)

        return img_tensor.numpy()
