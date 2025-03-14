import datetime
from kfp.v2 import dsl
from kfp import compiler

# Define pipeline components


@dsl.component(base_image="python:3.10", packages_to_install=["google-cloud-storage"])
def get_previous_model(bucket_name: str) -> str:
    """Fetch the previous model version from GCS."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)  # ✅ Fix: Removed "gs://"
    blobs = list(bucket.list_blobs(prefix="v-"))

    model_versions = sorted(
        [blob.name for blob in blobs if "v-" in blob.name],
        reverse=True
    )

    latest_model = model_versions[0] if model_versions else "None"
    print(f"Latest model version: {latest_model}")
    return latest_model


@dsl.component(base_image="python:3.10")
def train_model(
    model_version: str,
    previous_model_version: str,
    pipeline_root: str,
    project_id: str,
    region: str,
    training_image: str,
):
    """Executes a Vertex AI Custom Training Job using a container."""
    from google.cloud import aiplatform
    aiplatform.init(project=project_id, location=region,
                    staging_bucket=pipeline_root)

    train_job = aiplatform.CustomContainerTrainingJob(
        display_name="train-creature-model",
        container_uri=training_image,
    )

    train_job.run(
        model_display_name="creature-vision-model",
        args=[
            "--previous_model_version", previous_model_version,
            "--model_version", model_version
        ],
    )


@dsl.container_component
def run_dataflow(model_version: str, region: str, preprocessing_image: str):
    """Runs the Dataflow Flex Template with the given model version."""
    return dsl.ContainerSpec(
        # ✅ Fix: Ensure preprocessing_image is constant at compile time
        image=preprocessing_image,
        command=["gcloud"],
        args=[
            "dataflow", "flex-template", "run", "creature-vis-processing",
            "--template-file-gcs-location=gs://dataflow-use1/templates/creature-vision-template.json",
            "--region", region,  # ✅ Fix: Now correctly passed as an argument
            "--parameters=max_files=1000",
            "--parameters", f"model_version={model_version}",
        ],
    )


@dsl.container_component
def build_inference_image(model_output_path: str, model_version: str, inference_image: str):
    """Builds and pushes the new inference container image."""
    return dsl.ContainerSpec(
        image="gcr.io/cloud-builders/docker",
        command=["bash", "-c"],
        args=[
            "echo", "Building and pushing new inference container...",
            "&&",
            "docker", "build", "-t", inference_image,
            "--build-arg", f"MODEL_URI={model_output_path}",
            "--build-arg", f"MODEL_VERSION={model_version}",
            "&&",
            "docker", "push", inference_image
        ],
    )


@dsl.container_component
def deploy_cloud_run(region: str, inference_image: str):
    """Deploys the updated inference service to Cloud Run."""
    return dsl.ContainerSpec(
        image="gcr.io/google.com/cloudsdktool/cloud-sdk:latest",
        command=["gcloud"],
        args=[
            "run", "deploy", "dog-prediction-app",
            "--image", inference_image,
            "--region", region,
            "--platform", "managed",
            "--allow-unauthenticated",
        ],
    )


@dsl.pipeline(
    name="creature-vision-pipeline"
)
def creature_vision_pipeline(
    project_id: str,
    region: str,
    pipeline_root: str,
    model_bucket: str,
    inference_image: str,
    training_image: str,
    preprocessing_image: str,
):
    """Pipeline that orchestrates Dataflow preprocessing, training, and Cloud Run deployment."""

    # Define Model Version as a Timestamp
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    model_version = f"v-{date_str}"
    model_output_path = f"{model_bucket}/{model_version}/{model_version}.keras"

    # ✅ Fix: Pass pipeline parameters correctly
    get_previous_model_task = get_previous_model(bucket_name=model_bucket)
    dataflow_task = run_dataflow(
        model_version=model_version, region=region, preprocessing_image=preprocessing_image)

    train_model_task = train_model(
        model_version=model_version,
        previous_model_version=get_previous_model_task.output,
        pipeline_root=pipeline_root,
        project_id=project_id,
        region=region,
        training_image=training_image,
    ).after(get_previous_model_task, dataflow_task)

    build_inference_image_task = build_inference_image(
        model_output_path=model_output_path,
        model_version=model_version,
        inference_image=inference_image
    ).after(train_model_task)

    deploy_cloud_run_task = deploy_cloud_run(
        region=region, inference_image=inference_image
    ).after(build_inference_image_task)


# ✅ Fix: Ensure pipeline_root is explicitly set
compiler.Compiler().compile(
    pipeline_func=creature_vision_pipeline,
    package_path="creature_vision_pipeline.json"
)
