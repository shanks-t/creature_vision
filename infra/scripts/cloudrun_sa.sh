# Create the service account
gcloud iam service-accounts create cloudrun-gcs-sa \
    --description="Service account for Cloud Run to access GCS" \
    --display-name="Cloud Run GCS Service Account"

# Get your project ID
PROJECT_ID=$(gcloud config get-value project)

# Grant the service account permission to access GCS
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:cloudrun-gcs-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

# Optional: If you need write access to GCS, add this role as well
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:cloudrun-gcs-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectCreator"