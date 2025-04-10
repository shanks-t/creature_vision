import datetime
from kfp import dsl
from kfp import compiler
from google.cloud import aiplatform

# Component: Deploy Cloud Run Service


@dsl.container_component
def deploy_cloud_run_inference_service(
    project_id: str,
    region: str,
    service_name: str,
    image_uri: str,
    model_version: str,
):
    return dsl.ContainerSpec(
        image="gcr.io/google.com/cloudsdktool/cloud-sdk:slim",
        command=["gcloud", "run", "deploy", service_name],
        args=[
            "--image",
            image_uri,
            "--region",
            region,
            "--platform",
            "managed",
            "--project",
            project_id,
            "--allow-unauthenticated",
            "--set-env-vars",
            f"MODEL_VERSION={model_version}",
            "--memory",
            "2Gi",
            "--timeout",
            "300",
        ],
    )


# Pipeline that only runs the deploy step


@dsl.pipeline(
    name="test-cloudrun-deployment",
    description="Test deploying inference service to Cloud Run with a specific model version",
)
def test_cloudrun_pipeline(
    project_id: str, region: str, service_name: str, image_uri: str, model_version: str
):
    deploy_cloud_run_inference_service(
        project_id=project_id,
        region=region,
        service_name=service_name,
        image_uri=image_uri,
        model_version=model_version,
    )


# ---- Config & Compile ----
PROJECT_ID = "creature-vision"
REGION = "us-east1"
PIPELINE_ROOT = "gs://creature-vision-pipeline-artifacts"
SERVICE_NAME = "dog-predictor"
INFERENCE_IMAGE = (
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}/dog-prediction-app/inference:latest"
)
MODEL_VERSION = "mar-09-2025"

# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=test_cloudrun_pipeline,
    package_path="test_deploy_cloudrun_pipeline.json",
)

# Submit the pipeline job
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=PIPELINE_ROOT)

job = aiplatform.PipelineJob(
    display_name="test-cloudrun-deploy-job",
    template_path="test_deploy_cloudrun_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
    parameter_values={
        "project_id": PROJECT_ID,
        "region": REGION,
        "service_name": SERVICE_NAME,
        "image_uri": INFERENCE_IMAGE,
        "model_version": MODEL_VERSION,
    },
)

job.run(service_account=f"kubeflow-pipeline-sa@{PROJECT_ID}.iam.gserviceaccount.com")
