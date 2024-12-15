# tests/test_data_loader.py
import pytest
from unittest.mock import Mock, patch
from PIL import Image
import json
import os
import numpy as np
from preprocessing.data_loader import GCSDataLoader


class TestGCSDataLoader:
    @pytest.fixture
    def mock_storage_client(self):
        with patch('google.cloud.storage.Client') as mock_client:
            yield mock_client

    @pytest.fixture
    def mock_bucket(self, mock_storage_client):
        mock_bucket = Mock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        return mock_bucket

    @pytest.fixture
    def mock_data(self):
        # Create mock data structure
        os.makedirs('mock_data/correct_predictions', exist_ok=True)
        os.makedirs('mock_data/incorrect_predictions', exist_ok=True)

        # Create test image
        img = Image.new('RGB', (224, 224), color='blue')
        img.save('mock_data/correct_predictions/test_image.jpg')

        # Create test label
        label_data = {
            "model_label": "Bouvier_des_Flandres",
            "api_label": "affenpinscher"
        }
        with open('mock_data/correct_predictions/test_image_labels.json', 'w') as f:
            json.dump(label_data, f)

        return 'mock_data'

    def test_load_single_image_and_label(self, mock_bucket, mock_data):
        # Set up mock responses
        with open('mock_data/correct_predictions/test_image.jpg', 'rb') as f:
            mock_bucket.blob().download_as_bytes.return_value = f.read()
        with open('mock_data/correct_predictions/test_image_labels.json', 'r') as f:
            mock_bucket.blob().download_as_text.return_value = f.read()

        loader = GCSDataLoader(bucket_name='mock_bucket')
        image, label = loader.load_single_sample(
            'correct_predictions/test_image.jpg',
            'correct_predictions/test_image_labels.json'
        )

        assert image.shape == (224, 224, 3)
        assert isinstance(label, dict)
        assert label['api_label'] == 'affenpinscher'

    def test_batch_loading(self, mock_bucket, mock_data):
        # Set up mock responses
        mock_bucket.list_blobs.return_value = [
            Mock(name='correct_predictions/test_image.jpg'),
            Mock(name='incorrect_predictions/test_image2.jpg')
        ]
        with open('mock_data/correct_predictions/test_image.jpg', 'rb') as f:
            mock_bucket.blob().download_as_bytes.return_value = f.read()
        with open('mock_data/correct_predictions/test_image_labels.json', 'r') as f:
            mock_bucket.blob().download_as_text.return_value = f.read()

        loader = GCSDataLoader(bucket_name='mock_bucket')
        images, labels = loader.load_batch(batch_size=2)

        assert len(images) <= 2
        assert len(images) == len(labels)
        assert images[0].shape == (224, 224, 3)
