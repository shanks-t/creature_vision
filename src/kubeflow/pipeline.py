import datetime

from kfp import dsl
from kfp import compiler
from google.cloud import aiplatform as vertex_ai

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

# Step 1: Define Individual Components as Functions

# Initialize Vertex AI to use the correct region
vertex_ai.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=PIPELINE_ROOT  # Ensures consistency between Vertex AI & GCS
)


@dsl.component(base_image="gcr.io/google.com/cloudsdktool/cloud-sdk:latest")
def get_previous_model(bucket: str) -> str:
    """Fetch the previous model version from GCS."""
    import subprocess

    cmd = f"gsutil ls {bucket}/ | grep -o 'v-[0-9]*-[0-9]*' | sort | tail -n 1"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    latest_model = result.stdout.strip()
    print(f"Latest model version: {latest_model}")

    return latest_model


@dsl.container_component
def run_dataflow(model_version: str):
    """Runs the Dataflow Flex Template with the given model version."""
    return dsl.ContainerSpec(
        image="gcr.io/google.com/cloudsdktool/cloud-sdk:latest",
        command=["gcloud"],
        args=[
            "dataflow", "flex-template", "run", "creature-vis-processing",
            "--template-file-gcs-location=gs://dataflow-use1/templates/creature-vision-template.json",
            "--region=us-east1",
            "--parameters=max_files=1000",
            f"--parameters=model_version={model_version}"
        ]
    )


@dsl.component(base_image="python:3.10")
def train_model(
    model_version: str,
    previous_model_version: str,
    pipeline_root: str,
    project_id: str,
    region: str,
    training_image: str
):
    """Executes a Vertex AI Custom Training Job using a container."""
    from google.cloud import aiplatform

    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region,
                    staging_bucket=pipeline_root)

    # Create a Custom Training Job
    train_job = aiplatform.CustomContainerTrainingJob(
        display_name="train-creature-model",
        container_uri=training_image,
        model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-9:latest"
    )

    # Run Training
    train_job.run(
        model_display_name="creature-vision-model",
        args=[
            "--previous_model_version", previous_model_version,
            "--model_version", model_version
        ]
    )


@dsl.container_component
def build_inference_image(model_output_path: str, model_version: str):
    """Builds and pushes the new inference container image."""
    return dsl.ContainerSpec(
        image="gcr.io/cloud-builders/docker",
        command=["bash", "-c"],
        args=[
            f"""
            echo "Building and pushing new inference container..."
            docker build -t {INFERENCE_IMAGE} --build-arg MODEL_URI={model_output_path} --build-arg MODEL_VERSION={model_version} .
            docker push {INFERENCE_IMAGE}
            """
        ]
    )


@dsl.container_component
def deploy_cloud_run():
    """Deploys the updated inference service to Cloud Run."""
    return dsl.ContainerSpec(
        image="gcr.io/google.com/cloudsdktool/cloud-sdk:latest",
        command=["gcloud"],
        args=[
            "run", "deploy", "dog-prediction-app",
            "--image", INFERENCE_IMAGE,
            "--region", REGION,
            "--platform", "managed",
            "--allow-unauthenticated",
        ]
    )


# Step 2: Define the Pipeline Using Correct Task Dependencies
@dsl.pipeline(
    name="creature-vision-pipeline",
    pipeline_root=PIPELINE_ROOT,
)
def creature_vision_pipeline():
    """Pipeline that orchestrates Dataflow preprocessing, training, and Cloud Run deployment."""

    # Define Global Model Version
    date_str = datetime.datetime.now().strftime(
        "%Y%m%d-%H%M")  # Unique timestamp-based version
    model_version = f"v-{date_str}"
    model_output_path = f"{MODEL_BUCKET}/{model_version}/{model_version}.keras"

    # Step 1: Retrieve Previous Model Version
    get_previous_model_task = get_previous_model(bucket=MODEL_BUCKET)

    # Step 2: Run Dataflow with Model Version
    dataflow_task = run_dataflow(model_version=model_version)

    # Step 3: Train Model using the New Component
    train_model_task = train_model(
        model_version=model_version,
        previous_model_version=get_previous_model_task.output,
        pipeline_root=PIPELINE_ROOT,
        project_id=PROJECT_ID,
        region=REGION,
        training_image=TRAINING_IMAGE
    ).after(get_previous_model_task, dataflow_task)

    # Step 4: Build and Push New Inference Container
    build_inference_image_task = build_inference_image(
        model_output_path=model_output_path, model_version=model_version
    ).after(train_model_task)

    # Step 5: Deploy to Cloud Run
    deploy_cloud_run_task = deploy_cloud_run().after(build_inference_image_task)


# Step 3: Compile the Pipeline
compiler.Compiler().compile(
    pipeline_func=creature_vision_pipeline,
    package_path="creature_vision_pipeline.json"
)
