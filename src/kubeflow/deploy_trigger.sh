gcloud functions deploy trigger-pipeline \
    --runtime python39 \
    --region us-east1 \
    --entry-point trigger_pipeline \
    --trigger-http \
    --allow-unauthenticated
