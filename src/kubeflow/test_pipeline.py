import datetime
import uuid
from kfp import dsl
from kfp import compiler
from google.cloud import aiplatform
from google_cloud_pipeline_components.v1.dataflow import DataflowFlexTemplateJobOp

# Define pipeline components


@dsl.component(base_image="python:3.10", packages_to_install=["google-cloud-storage"])
def get_previous_model(bucket_name: str) -> str:
    """Fetch the previous model version from GCS."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix="v-"))

    model_versions = sorted(
        [blob.name for blob in blobs if "v-" in blob.name],
        reverse=True
    )

    latest_model = model_versions[0] if model_versions else "None"
    print(f"Latest model version: {latest_model}")
    return latest_model


@dsl.pipeline(
    name="creature-vision-pipeline",
    description="Pipeline that orchestrates Dataflow preprocessing, training, and Cloud Run deployment."
)
def creature_vision_pipeline(
    project_id: str,
    region: str,
    pipeline_root: str,
    model_bucket: str,
    inference_image: str,
    training_image: str,
    service_account: str,
    gcs_template_path: str
):
    """Kubeflow Pipeline that runs a Dataflow Flex Template job after retrieving the previous model."""
    date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    model_version = f"v-{date_str}"

    # Task to get the previous model version
    get_previous_model_task = get_previous_model(bucket_name=model_bucket)

    # Define the Dataflow task
    dataflow_task = DataflowFlexTemplateJobOp(
        location=region,
        container_spec_gcs_path=gcs_template_path,
        job_name=f"creature-vis-training",
        parameters={
            "version": get_previous_model_task.output,
            "max_files": "100"
        },
        service_account_email=service_account
    )


# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=creature_vision_pipeline,
    package_path="creature_vision_pipeline.json"
)

# Project Configuration
PROJECT_ID = "creature-vision"
REGION = "us-east1"
PIPELINE_ROOT = f"gs://creature-vision-pipeline-artifacts"
SERVICE_ACCOUNT = f"kubeflow-pipeline-sa@{PROJECT_ID}.iam.gserviceaccount.com"
GCS_TEMPLATE_PATH = "gs://dataflow-use1/templates/creature-vision-template.json"

# Artifact Registry URIs
ARTIFACT_REGISTRY = f"{REGION}-docker.pkg.dev/{PROJECT_ID}"
INFERENCE_IMAGE = f"{ARTIFACT_REGISTRY}/dog-prediction-app/inference:latest"
TRAINING_IMAGE = f"{ARTIFACT_REGISTRY}/creature-vis-training/training:latest"

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
    "service_account": SERVICE_ACCOUNT,
    "gcs_template_path": GCS_TEMPLATE_PATH
}

# Create and run the pipeline job
pipeline_job = aiplatform.PipelineJob(
    display_name="creature-vision-pipeline-job",
    template_path="creature_vision_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
    parameter_values=parameter_values,
    enable_caching=True,
)

pipeline_job.run(service_account=SERVICE_ACCOUNT)
