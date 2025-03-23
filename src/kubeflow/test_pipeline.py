import time
import datetime

from kfp import dsl
from kfp import compiler


from google.cloud import aiplatform
from google_cloud_pipeline_components.v1.dataflow import DataflowFlexTemplateJobOp
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp


# Simplified get_previous_model component that returns a string
@dsl.component(base_image="python:3.10", packages_to_install=["google-cloud-storage"])
def get_previous_model(bucket_name: str) -> str:
    """Fetch the previous model version from GCS and return it as a string."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix="v-"))

    model_versions = sorted(
        [blob.name for blob in blobs if "v-" in blob.name],
        reverse=True
    )

    latest_model = model_versions[0] if model_versions else "None"
    return latest_model  # Directly returning the value


@dsl.component(base_image="python:3.10", packages_to_install=["google-api-python-client"])
def wait_for_dataflow_job(project_id: str, region: str, job_name: str, poll_interval: int = 30, timeout: int = 1800):
    """
    Polls the Dataflow job status every `poll_interval` seconds until it reaches a terminal state
    or the `timeout` is reached.
    """
    from googleapiclient.discovery import build

    dataflow = build('dataflow', 'v1b3')
    elapsed_time = 0
    job_id = None

    # Retrieve the job ID based on the job name
    request = dataflow.projects().locations().jobs().list(
        projectId=project_id, location=region)
    while request is not None:
        response = request.execute()
        for job in response.get('jobs', []):
            if job['name'] == job_name:
                job_id = job['id']
                break
        request = dataflow.projects().locations().jobs().list_next(
            previous_request=request, previous_response=response)
        if job_id:
            break

    if not job_id:
        raise ValueError(f"No job found with name {job_name}")

    while elapsed_time < timeout:
        job = dataflow.projects().locations().jobs().get(
            projectId=project_id,
            location=region,
            jobId=job_id
        ).execute()

        state = job.get('currentState', 'UNKNOWN')
        print(f"Dataflow job {job_id} is in state: {state}")

        if state == 'JOB_STATE_DONE':
            print(f"Dataflow job {job_id} completed successfully.")
            return
        elif state in ['JOB_STATE_FAILED', 'JOB_STATE_CANCELLED']:
            raise RuntimeError(
                f"Dataflow job {job_id} failed with state: {state}")

        time.sleep(poll_interval)
        elapsed_time += poll_interval

    raise TimeoutError(
        f"Dataflow job {job_id} did not complete within the allotted time of {timeout} seconds.")


# Pipeline definition
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
    python_package_gcs_uri: str,
    service_account: str,
    gcs_template_path: str
):
    """Kubeflow Pipeline that runs a Dataflow Flex Template job after retrieving the previous model."""
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    model_version = f"v-{date_str}"

    # Task to get the previous model version
    get_previous_model_task = get_previous_model(bucket_name=model_bucket)

   # Define the Dataflow task
    df_job_name = f"creature-vis-training-{date_str}"
    dataflow_task = DataflowFlexTemplateJobOp(
        location=region,
        container_spec_gcs_path=gcs_template_path,
        job_name=df_job_name,
        parameters={
            "version": model_version,
            "max_files": "1000"
        },
        service_account_email=service_account,
        launch_options={"enable_preflight_validation": "false"},
    )

    # Wait for Dataflow job to complete
    wait_task = wait_for_dataflow_job(
        project_id=project_id,
        region=region,
        job_name=df_job_name
    )

    training_task = CustomTrainingJobOp(
        display_name="creature-vision-training",
        project=project_id,
        location=region,
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": "e2-standard-4",
                },
                "replica_count": 1,
                "python_package_spec": {
                    "executor_image_uri": "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-16.py310:latest",
                    "package_uris": [python_package_gcs_uri],
                    "python_module": "creature_vision_training.main",
                    "args": [
                        "--version", model_version,
                        "--previous_model_version", get_previous_model_task.output
                    ],
                },
            }
        ],
        service_account=service_account
    ).after(wait_task)


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


# GCS Artifacts
MODEL_BUCKET = "tf_models_cv"
PYTHON_PACKAGE_URI = "gs://creture-vision-ml-artifacts/python_packages/creature_vision_training-0.1.tar.gz"

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
    "python_package_gcs_uri": PYTHON_PACKAGE_URI,
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
