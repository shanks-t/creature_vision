from google.cloud import aiplatform

# Project Configuration
PROJECT_ID = "creature-vision"
REGION = "us-east1"
PIPELINE_ROOT = f"gs://creature-vision-pipeline-artifacts"
SERVICE_ACCOUNT = f"kubeflow-pipeline-sa@{PROJECT_ID}.iam.gserviceaccount.com"

# Artifact Registry URIs
ARTIFACT_REGISTRY = f"{REGION}-docker.pkg.dev/{PROJECT_ID}"
INFERENCE_IMAGE = f"{ARTIFACT_REGISTRY}/dog-prediction-app/inference:latest"
TRAINING_IMAGE = f"{ARTIFACT_REGISTRY}/creature-vis-training/training:latest"
PREPROCESSING_IMAGE = f"{ARTIFACT_REGISTRY}/creature-vis-preprocessing/preprocessing:latest"

# Model Storage in GCS
MODEL_BUCKET = "tf_models_cv"

# Initialize Vertex AI
aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=PIPELINE_ROOT
)

# Define pipeline parameters
parameter_values = {
    "project_id": PROJECT_ID,
    "region": REGION,
    "pipeline_root": PIPELINE_ROOT,
    "model_bucket": MODEL_BUCKET,
    "inference_image": INFERENCE_IMAGE,
    "training_image": TRAINING_IMAGE,
    "preprocessing_image": PREPROCESSING_IMAGE,
    "service_account": SERVICE_ACCOUNT
}

# Create and run the pipeline job
pipeline_job = aiplatform.PipelineJob(
    display_name="creature-vision-pipeline-job",
    template_path="creature_vision_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
    parameter_values=parameter_values,
    enable_caching=False,
)

pipeline_job.run(service_account=SERVICE_ACCOUNT)
