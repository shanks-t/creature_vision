from google.cloud import aiplatform
import functions_framework
import datetime


@functions_framework.http
def trigger_pipeline(request):
    project = "creature-vision"
    region = "us-east1"
    pipeline_root = "gs://creature-vision-pipeline-artifacts"
    template_path = "gs://creature-vision-pipeline-artifacts/kubeflow-templates/creature_vision_pipeline.json"

    aiplatform.init(project=project, location=region,
                    staging_bucket=pipeline_root)

    date_str = datetime.datetime.now().strftime("%Y%m%d%H")
    model_version = f"v-{date_str}"

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
        "max_files": "1200",
    }

    job = aiplatform.PipelineJob(
        display_name=f"creature-vision-pipeline-job-{date_str}",
        template_path=template_path,
        pipeline_root=pipeline_root,
        parameter_values=parameter_values,
        enable_caching=False,
    )

    # Start job and don't wait for it to finish
    job.submit(service_account=parameter_values["service_account"])

    return {
        "message": "Pipeline job submitted successfully",
        "job_name": job.display_name,
        "create_time": str(job.create_time),
        "state": job.state.name,
        "pipeline_url": f"https://console.cloud.google.com/vertex-ai/locations/{region}/pipelines/runs/{job.resource_name.split('/')[-1]}?project={project}"
    }, 200
