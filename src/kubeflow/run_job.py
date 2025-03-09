from google.cloud import aiplatform

# Project Configuration
PROJECT_ID = "creature-vision"
REGION = "us-east1"
PIPELINE_ROOT = f"gs://creature-vision-pipeline-artifacts"

# Artifact Registry URIs (from Makefile)
ARTIFACT_REGISTRY = f"{REGION}-docker.pkg.dev/{PROJECT_ID}"
INFERENCE_IMAGE = f"{ARTIFACT_REGISTRY}/dog-prediction-app/inference:latest"
TRAINING_IMAGE = f"{ARTIFACT_REGISTRY}/creature-vis-training/training:latest"
PREPROCESSING_IMAGE = f"{ARTIFACT_REGISTRY}/creature-vis-preprocessing/preprocessing:latest"

# Model Storage in GCS
MODEL_BUCKET = "gs://tf_models_cv"

# Initialize Vertex AI
aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=PIPELINE_ROOT
)

# Define pipeline parameters
parameter_values = {
}

# Create and run the pipeline job
pipeline_job = aiplatform.PipelineJob(
    display_name="creature-vision-pipeline-job",
    template_path="creature_vision_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
    parameter_values=parameter_values,
)

pipeline_job.run()
