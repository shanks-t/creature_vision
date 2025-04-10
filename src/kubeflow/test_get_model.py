from kfp import compiler
from kfp import dsl
from kfp.dsl import component
from google.cloud import aiplatform


@component(base_image="python:3.9")
def get_cloud_run_model_version(
    project_id: str,
    region: str,
    service_name: str,
) -> str:
    import subprocess
    import json

    cmd = [
        "gcloud",
        "run",
        "services",
        "describe",
        service_name,
        "--platform",
        "managed",
        "--project",
        project_id,
        "--region",
        region,
        "--format",
        "json",
    ]

    result = subprocess.run(cmd, capture_output=True, check=True, text=True)
    service_config = json.loads(result.stdout)
    env_vars = service_config["spec"]["template"]["spec"]["containers"][0]["env"]

    for var in env_vars:
        if var["name"] == "MODEL_VERSION":
            return var["value"]

    return "unknown"


@dsl.pipeline
def retrieve_versions_pipeline(
    project_id: str,
    region: str,
    champion_service: str,
    challenger_service: str,
):
    champion_task = get_cloud_run_model_version(
        project_id=project_id,
        region=region,
        service_name=champion_service,
    )

    challenger_task = get_cloud_run_model_version(
        project_id=project_id,
        region=region,
        service_name=challenger_service,
    )


# ---- Config & Compile ----
PROJECT_ID = "creature-vision"
REGION = "us-east1"
PIPELINE_ROOT = "gs://creature-vision-pipeline-artifacts"

# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=retrieve_versions_pipeline,
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
        "champion_service": "inference-champion",
        "challenger_service": "inference-challenger",
    },
)

job.run(service_account=f"kubeflow-pipeline-sa@{PROJECT_ID}.iam.gserviceaccount.com")
