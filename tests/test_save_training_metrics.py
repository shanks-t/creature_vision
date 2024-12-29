from src.training.model import save_model_training_metrics


def test_save_training_metrics(mocker):
    # Mock BigQuery client
    mock_bq = mocker.patch('google.cloud.bigquery.Client')
    mock_client = mock_bq.return_value
    mock_client.insert_rows_json.return_value = []

    # Test metrics save
    save_model_training_metrics(
        accuracy=0.85,
        top_k_accuracy=0.95,
        model_version="m_net_s_v3.1"
    )

    # Verify BigQuery interaction
    mock_client.insert_rows_json.assert_called_once()
    args = mock_client.insert_rows_json.call_args[0]
    assert args[0] == "dog-prediction-app.training_metrics"
    assert len(args[1]) == 1
    assert args[1][0]["model_version"] == "m_net_s_v3.1"
    assert args[1][0]["accuracy"] == 0.85
    assert args[1][0]["top_k_accuracy"] == 0.95
