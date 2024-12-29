from training.model import create_model
from training.model import save_model
import os


def test_save_model(mocker):
    # Mock GCS client and related components
    mock_storage = mocker.patch('google.cloud.storage.Client')
    mock_bucket = mocker.MagicMock()
    mock_blob = mocker.MagicMock()

    # Configure mock chain
    mock_storage.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    # Create test model
    model = create_model(num_classes=2)

    # Test save function
    save_model(model, version="test_v1", bucket_name="test-bucket")

    # Verify local save
    assert os.path.exists("models/test_v1.keras")

    # Verify GCS upload
    mock_bucket.blob.assert_called_once_with("test_v1.keras")
    mock_blob.upload_from_filename.assert_called_once_with(
        "models/test_v1.keras")
