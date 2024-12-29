# tests/test_image_processor.py
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from src.preprocessing.image_processor import ImageProcessor
import tensorflow as tf


class TestImageProcessor:
    @pytest.fixture
    def data_dir(self):
        return Path("./data")

    @pytest.fixture
    def sample_image(self, data_dir):
        # Get first image from correct_predictions directory
        image_path = next(data_dir.joinpath(
            "correct_predictions").glob("*.jpg"))

        image = Image.open(image_path)
        return np.array(image)

    @pytest.fixture
    def processor(self):
        return ImageProcessor()

    # def test_preprocess_batch(self, processor, data_dir):
    #     image_paths = list(data_dir.joinpath(
    #         "correct_predictions").glob("*.jpg"))[:50]

    #     images = [np.array(Image.open(path)) for path in image_paths]
    #     batch = np.stack(images)

    #     processed = processor.preprocess_batch(batch)
    #     assert processed.shape == (50, 224, 224, 3)
    #     assert processed.dtype == np.float32
    #     assert -1.0 <= processed.min() <= processed.max() <= 1.0

    def test_preprocess_batch(self, processor, data_dir):
        image_paths = list(data_dir.joinpath(
            "correct_predictions").glob("*.jpg"))[:50]

        images = [np.array(Image.open(path)) for path in image_paths]
        batch = np.stack(images)

        processed = processor.preprocess_batch(batch)
        assert processed.shape == (50, 224, 224, 3)
        assert processed.dtype == tf.float32
        assert - \
            1.0 <= tf.reduce_min(processed) <= tf.reduce_max(processed) <= 1.0
