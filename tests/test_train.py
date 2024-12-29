# test_train.py
import pytest
import tensorflow as tf
from src.training.dataset import create_training_dataset


def test_create_training_dataset(mocker):
    # Mock the GCS client
    mock_storage = mocker.patch('google.cloud.storage.Client')
    mock_bucket = mocker.MagicMock()
    mock_blob = mocker.MagicMock()

    # Configure mock responses
    mock_storage.return_value.bucket.return_value = mock_bucket
    mock_bucket.list_blobs.return_value = [
        mocker.MagicMock(name='correct_predictions/test_image.jpg')
    ]
    mock_bucket.blob.return_value = mock_blob

    # Set up mock blob responses using actual test files
    with open('mock_data/correct_predictions/test_image.jpg', 'rb') as img_file:
        mock_blob.download_as_bytes.return_value = img_file.read()
    with open('mock_data/correct_predictions/test_image_labels.json', 'r') as label_file:
        mock_blob.download_as_text.return_value = label_file.read()

    # Run the test
    dataset, num_classes = create_training_dataset(
        bucket_name="test-bucket",
        num_examples=32,
        batch_size=16
    )

    # Verify the mocks were called
    mock_bucket.list_blobs.assert_called()
    mock_blob.download_as_bytes.assert_called()
    mock_blob.download_as_text.assert_called()
    # Check first batch
    for images, labels in dataset.take(1):
        assert images.shape[0] <= 16  # batch size
        assert images.shape[-1] == 3  # RGB channels
        assert labels.shape[0] == images.shape[0]
