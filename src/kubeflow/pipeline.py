from kfp import dsl
from kfp.v2 import compiler
from google_cloud_pipeline_components import aiplatform as vertex_ai

import datetime

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


@dsl.pipeline(
    name="creature-vision-pipeline",
    pipeline_root=PIPELINE_ROOT,
)
def creature_vision_pipeline():
    """Pipeline that orchestrates Dataflow preprocessing, training, and Cloud Run deployment."""

    # Step 1: Run Dataflow Flex Template for Image Preprocessing
    dataflow_task = dsl.ContainerOp(
        name="run-dataflow",
        image="gcr.io/google.com/cloudsdktool/cloud-sdk:latest",
        command=["gcloud"],
        arguments=[
            "dataflow", "flex-template", "run", "creature-vis-processing",
            "--template-file-gcs-location=gs://dataflow-use1/templates/creature-vision-template.json",
            "--region=us-east1",
            "--parameters=max_files=1000"
        ]
    )

    # Generate the date-based model path
    date_str = datetime.datetime.now().strftime("%b-%d-%Y").lower()
    model_output_path = f"{MODEL_BUCKET}/{date_str}/{date_str}.keras"

    # Step 2: Train TensorFlow Model using Vertex AI Custom Training
    train_model_task = vertex_ai.CustomContainerTrainingJobRunOp(
        display_name="train-creature-model",
        container_uri=TRAINING_IMAGE,  # Training container from Makefile
        model_display_name="creature-vision-model",
        staging_bucket=PIPELINE_ROOT,
        args=[
            "--model_gcs_path", model_output_path  # Pass new model path
        ]
    ).after(dataflow_task)

    # Step 3: Build and Push New Inference Container with the Updated Model
    build_inference_image_task = dsl.ContainerOp(
        name="build-inference-image",
        image="gcr.io/cloud-builders/docker",
        command=["bash", "-c"],
        arguments=[
            f"""
            echo "Building and pushing new inference container..."
            MODEL_URI={model_output_path}
            docker build -t {INFERENCE_IMAGE} --build-arg MODEL_URI=$MODEL_URI .
            docker push {INFERENCE_IMAGE}
            """
        ]
    ).after(train_model_task)

    # Step 4: Deploy the New Inference Service to Cloud Run
    deploy_cloud_run_task = dsl.ContainerOp(
        name="deploy-cloud-run",
        image="gcr.io/google.com/cloudsdktool/cloud-sdk:latest",
        command=["gcloud"],
        arguments=[
            "run", "deploy", "dog-prediction-app",
            "--image", INFERENCE_IMAGE,
            "--region", REGION,
            "--platform", "managed",
            "--allow-unauthenticated"
        ]
    ).after(build_inference_image_task)


# Compile pipeline
compiler.Compiler().compile(
    pipeline_func=creature_vision_pipeline,
    package_path="creature_vision_pipeline.json"
)
