from google.cloud import aiplatform, storage
import functions_framework
import json

VERSION_FILE = "gs://creature-vision-pipeline-artifacts/version.txt"
BASE_VERSION = "3_0"


def read_version_from_gcs(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        return BASE_VERSION

    return blob.download_as_text().strip()


def write_version_to_gcs(bucket_name, blob_name, version):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(version)


def bump_minor_version(version: str) -> str:
    major, minor = version.split("_")
    return f"{major}_{int(minor) + 1}"


@functions_framework.http
def trigger_pipeline(request):
    project = "creature-vision"
    region = "us-east1"
    pipeline_root = "gs://creature-vision-pipeline-artifacts"
    template_path = f"{pipeline_root}/kubeflow-templates/creature_vision_pipeline.json"

    # Parse GCS path
    version_bucket = VERSION_FILE.replace("gs://", "").split("/")[0]
    version_blob = "/".join(VERSION_FILE.replace("gs://", "").split("/")[1:])

    # Read and bump version
    current_version = read_version_from_gcs(version_bucket, version_blob)
    new_version = bump_minor_version(current_version)
    write_version_to_gcs(version_bucket, version_blob, new_version)

    aiplatform.init(project=project, location=region,
                    staging_bucket=pipeline_root)

    # Use bumped version
    model_version = f"v{new_version}"

    default_max_files = "1200"
    max_files = default_max_files

    if request.method == 'POST':
        try:
            request_json = request.get_json(silent=True)
            if request_json and 'max_files' in request_json:
                max_files = str(request_json['max_files'])
        except Exception:
            pass
    elif request.method == 'GET':
        max_files = request.args.get('max_files', default_max_files)

    parameter_values = {
        "project_id": project,
        "region": region,
        "pipeline_root": pipeline_root,
        "model_bucket": "tf_models_cv",
        "inference_image": f"{region}-docker.pkg.dev/{project}/dog-prediction-app/inference:latest",
        "inference_service": "dog-predictor",
        "python_package_gcs_uri": "gs://creture-vision-ml-artifacts/python_packages/creature_vision_training-0.1.tar.gz",
        "service_account": f"kubeflow-pipeline-sa@{project}.iam.gserviceaccount.com",
        "gcs_template_path": "gs://dataflow-use1/templates/creature-vision-template.json",
        "model_version": model_version,
        "max_files": max_files,
    }

    job = aiplatform.PipelineJob(
        display_name=f"creature-vision-pipeline-job-{model_version}",
        template_path=template_path,
        pipeline_root=pipeline_root,
        parameter_values=parameter_values,
        enable_caching=False,
    )

    job.submit(service_account=parameter_values["service_account"])

    return {
        "message": "Pipeline job submitted successfully",
        "job_name": job.display_name,
        "model_version": model_version,
        "state": job.state.name,
        "max_files": max_files,
        "pipeline_url": f"https://console.cloud.google.com/vertex-ai/locations/{region}/pipelines/runs/{job.resource_name.split('/')[-1]}?project={project}"
    }, 200
