gcloud scheduler jobs create http trigger-ml-pipeline \
    --schedule="0 2 * * *" \
    --uri="https://us-east1-trigger-pipeline-xyz.cloudfunctions.net/trigger-pipeline" \
    --http-method=POST \
    --oauth-service-account-email=<YOUR-SERVICE-ACCOUNT>
