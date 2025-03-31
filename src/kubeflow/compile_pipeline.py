import time
import datetime

from kfp import dsl
from kfp import compiler

from google.cloud import aiplatform
from google_cloud_pipeline_components.v1.dataflow import DataflowFlexTemplateJobOp
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp


@dsl.component(base_image="python:3.10", packages_to_install=["google-cloud-storage"])
def get_previous_model(bucket_name: str) -> str:
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix="v"))

    model_versions = sorted(
        [blob.name for blob in blobs if "v" in blob.name],
        reverse=True
    )

    latest_model = model_versions[0] if model_versions else "None"
    return latest_model


@dsl.component(base_image="python:3.10", packages_to_install=["google-api-python-client"])
def wait_for_dataflow_job(project_id: str, region: str, job_name: str, poll_interval: int = 30, timeout: int = 1800):
    from googleapiclient.discovery import build
    import time

    dataflow = build('dataflow', 'v1b3')
    elapsed_time = 0
    job_id = None

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
            "--image", image_uri,
            "--region", region,
            "--platform", "managed",
            "--project", project_id,
            "--allow-unauthenticated",
            "--set-env-vars", f"MODEL_VERSION={model_version}",
            "--memory", "2Gi",
            "--timeout", "300",
        ]
    )


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
    inference_service: str,
    python_package_gcs_uri: str,
    service_account: str,
    gcs_template_path: str,
    model_version: str,
    max_files: str = "1000"
):

    get_previous_model_task = get_previous_model(bucket_name=model_bucket)

    df_job_name = f"creature-vis-training-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
    dataflow_task = DataflowFlexTemplateJobOp(
        location=region,
        container_spec_gcs_path=gcs_template_path,
        job_name=df_job_name,
        parameters={
            "version": model_version,
            "max_files": max_files
        },
        service_account_email=service_account,
        launch_options={"enable_preflight_validation": "false"},
    )

    wait_task = wait_for_dataflow_job(
        project_id=project_id,
        region=region,
        job_name=df_job_name
    ).after(dataflow_task)

    training_task = CustomTrainingJobOp(
        display_name="creature-vision-training",
        project=project_id,
        location=region,
        tensorboard="projects/284159624099/locations/us-east1/tensorboards/3018185806524186624",
        base_output_directory=f"{pipeline_root}/training-outputs/{model_version}",
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": "e2-standard-4",
                },
                "replica_count": 1,
                "python_package_spec": {
                    "executor_image_uri": "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest",
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

    deploy_task = deploy_cloud_run_inference_service(
        project_id=project_id,
        region=region,
        service_name=inference_service,
        image_uri=inference_image,
        model_version=model_version
    ).after(training_task)


compiler.Compiler().compile(
    pipeline_func=creature_vision_pipeline,
    package_path="creature_vision_pipeline.json"
)
