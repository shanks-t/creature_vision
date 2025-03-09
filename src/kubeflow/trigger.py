import functions_framework
import google.auth
import requests

PROJECT_ID = "creature-vision"
REGION = "us-east1"
PIPELINE_NAME = "creature-vision-pipeline"


@functions_framework.http
def trigger_pipeline(request):
    """Cloud Function that triggers the Vertex AI Pipeline."""
    credentials, _ = google.auth.default()
    access_token = credentials.token

    url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/pipelines/{PIPELINE_NAME}:run"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    data = {
        "pipelineSpec": {},
        "runtimeConfig": {}
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return f"Pipeline {PIPELINE_NAME} triggered successfully!", 200
    else:
        return f"Failed to trigger pipeline: {response.text}", 500
