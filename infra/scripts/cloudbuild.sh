#!/bin/bash

# Set your project ID
export PROJECT_ID="creature-vision"

# Build images
# docker compose -f docker-compose.cloudrun.yaml build

# Tag images
# docker tag dog-prediction-app:latest gcr.io/${PROJECT_ID}/dog-prediction-app:latest
docker tag tensorflow/serving:latest gcr.io/${PROJECT_ID}/tfserving:latest

# # Authenticate with Google Cloud
# gcloud auth configure-docker

# Push images to GCR
docker push gcr.io/${PROJECT_ID}/dog-prediction-app:latest
docker push gcr.io/${PROJECT_ID}/tfserving:latest

# Deploy to Cloud Run
# gcloud run deploy dog-prediction-app \
#   --image gcr.io/${PROJECT_ID}/dog-prediction-app:latest \
#   --platform managed \
#   --region us-east1 \
#   --memory 2Gi \
#   --allow-unauthenticated

gcloud run deploy tfserving \
  --image gcr.io/${PROJECT_ID}/tfserving:latest \
  --port 8501 \
  --set-env-vars MODEL_CONFIG_FILE=/models/models.config \
  --platform managed \
  --region us-east1 \
  --service-account cloudrun-gcs-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --memory 2Gi \
  --allow-unauthenticated