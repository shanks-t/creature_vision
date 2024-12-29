# tests/test_data_loader.py
import pytest
from PIL import Image
import json


@pytest.fixture
def mock_gcs_client(mocker):
    """Setup GCS mocks using existing mock data"""
    # Mock the storage client before any GCS operations
    mock_storage = mocker.patch('google.cloud.storage.Client')
    mock_bucket = mocker.MagicMock()
    mock_blob = mocker.MagicMock()

    # Read actual test files
    with open('mock_data/correct_predictions/test_image.jpg', 'rb') as img_file:
        mock_blob.download_as_bytes.return_value = img_file.read()
    with open('mock_data/correct_predictions/test_image_labels.json', 'r') as label_file:
        mock_blob.download_as_text.return_value = label_file.read()

    # Configure mock chain
    mock_storage.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    mock_bucket.list_blobs.return_value = [
        mocker.MagicMock(name='correct_predictions/test_image.jpg')
    ]

    return mock_storage, mock_bucket, mock_blob


def test_load_single_sample(mock_gcs_client):
    """Test loading a single image and label"""
    from src.preprocessing.data_loader import GCSDataLoader

    _, mock_bucket, mock_blob = mock_gcs_client
    loader = GCSDataLoader('test-bucket')

    image, label = loader.load_single_sample(
        'correct_predictions/test_image.jpg',
        'correct_predictions/test_image_labels.json'
    )

    # Verify mock interactions
    mock_bucket.blob.assert_any_call('correct_predictions/test_image.jpg')
    mock_blob.download_as_bytes.assert_called_once()
    mock_blob.download_as_text.assert_called_once()

    # Verify returned data
    assert image.shape == (224, 224, 3)
    assert 'api_label' in label


def test_load_batch(mock_gcs_client, mocker):
    """Test loading a batch of images and labels"""
    from preprocessing.data_loader import GCSDataLoader

    _, mock_bucket, mock_blob = mock_gcs_client
    loader = GCSDataLoader('test-bucket')

    images, labels = loader.load_batch(batch_size=1)

    # Verify mock interactions
    assert mock_bucket.list_blobs.call_count == 2

    mock_bucket.list_blobs.assert_has_calls([
        mocker.call(prefix='correct_predictions/'),
        mocker.call(prefix='incorrect_predictions/')
    ])

    assert mock_blob.download_as_bytes.called
    assert mock_blob.download_as_text.called

    # Verify returned data
    assert len(images) == 1
    assert len(labels) == 1
    assert images[0].shape == (224, 224, 3)
    assert 'api_label' in labels[0]
