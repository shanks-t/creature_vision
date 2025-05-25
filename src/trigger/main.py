import json
import datetime
import uuid

from google.cloud import storage, bigquery, aiplatform

# === CONFIGURATION ===
GCS_BUCKET = "ml_challenger_state"
GCS_STATE_FILE = "model_versions.json"
BQ_METRICS_TABLE = "creature-vision.dog_prediction_app.prediction_metrics"
PIPELINE_ROOT = "gs://creature-vision-pipeline-artifacts"
PIPELINE_JOB_NAME = f"cv-training-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
PIPELINE_JSON_URI = "gs://creature-vision-pipeline-artifacts/kubeflow-templates/creature_vision_pipeline.json"

# Replace with your actual values
PROJECT_ID = "creature-vision"
REGION = "us-east1"
SERVICE_ACCOUNT = "kubeflow-pipeline-sa@creature-vision.iam.gserviceaccount.com"
INFERENCE_IMAGE = (
    "us-east1-docker.pkg.dev/creature-vision/dog-prediction-app/inference:latest"
)
PYTHON_PACKAGE_GCS_URI = "gs://creture-vision-ml-artifacts/python_packages/creature_vision_training-0.1.tar.gz"
GCS_TEMPLATE_PATH = "gs://dataflow-use1/templates/creature-vision-template.json"


# === LOGIC ===
def load_model_versions():
    client = storage.Client()
    blob = client.bucket(GCS_BUCKET).blob(GCS_STATE_FILE)
    data = json.loads(blob.download_as_text())
    return data, blob


def evaluate_models(champion_version: str, challenger_version: str) -> str:
    bq = bigquery.Client(project=PROJECT_ID)
    query = f"""
        SELECT model_version, AVG(CAST(is_correct AS INT64)) AS avg_prediction_accuracy
        FROM `{BQ_METRICS_TABLE}`
        WHERE model_version IN ('{champion_version}', '{challenger_version}')
        GROUP BY model_version
    """
    results = {
        row.model_version: row.avg_prediction_accuracy
        for row in bq.query(query).result()
    }
    print("Model accuracies:", results)

    champ_score = results.get(champion_version, 0.0)
    chall_score = results.get(challenger_version, 0.0)

    return challenger_version if chall_score > champ_score else champion_version


def update_model_versions(blob, data, winner: str, model_version: str):
    if data["champion"]["model_version"] != winner:
        print(f"Promoting {winner} to champion")
        data["champion"] = {
            "model_version": winner,
            "deployed_at": datetime.datetime.now().isoformat() + "Z",
        }

        current_version = data["challenger"]["model_version"]
        version_parts = current_version.strip("v").split("_")
        major, minor = map(int, version_parts)

        new_challenger_version = f"v{major}_{minor + 1}"
        print(f"New challenger: {new_challenger_version}")
        data["challenger"] = {
            "model_version": new_challenger_version,
            "deployed_at": None,
        }
        blob.upload_from_string(json.dumps(data, indent=2))
        return winner, "champion-service", new_challenger_version
    else:
        current_version = data["challenger"]["model_version"]
        version_parts = current_version.strip("v").split("_")
        major, minor = map(int, version_parts)

        new_challenger_version = f"v{major}_{minor + 1}"
        print(f"New challenger: {new_challenger_version}")
        data["challenger"] = {
            "model_version": new_challenger_version,
            "deployed_at": None,
        }
        blob.upload_from_string(json.dumps(data, indent=2))
        return (
            current_version,
            "challenger-service",
            new_challenger_version,
        )


def trigger_pipeline(
    service_to_update,
    new_model_version,
    previous_model_version,
    use_caching,
):
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=PIPELINE_ROOT)

    job = aiplatform.PipelineJob(
        display_name="creature-vision-pipeline",
        template_path=PIPELINE_JSON_URI,
        job_id=PIPELINE_JOB_NAME,
        parameter_values={
            "project_id": PROJECT_ID,
            "region": REGION,
            "pipeline_root": PIPELINE_ROOT,
            "inference_image": INFERENCE_IMAGE,
            "python_package_gcs_uri": PYTHON_PACKAGE_GCS_URI,
            "service_account": SERVICE_ACCOUNT,
            "gcs_template_path": GCS_TEMPLATE_PATH,
            "model_version": new_model_version,
            "previous_model_version": previous_model_version,
            "service_to_update": service_to_update,
        },
        enable_caching=use_caching,
    )
    job.submit(service_account=SERVICE_ACCOUNT)
    return job


def run_pipeline_trigger(request):
    print("Triggered pipeline orchestration via Cloud Function.")

    use_caching = False  # default

    if request.method == "POST":
        try:
            request_json = request.get_json(silent=True)
            if request_json:
                use_caching = bool(request_json.get("use_caching", use_caching))
        except Exception as e:
            print(f"Error parsing request: {e}")
    elif request.method == "GET":
        use_caching = request.args.get("use_caching", "false").lower() == "true"

    # Step 1: Load model versions
    data, blob = load_model_versions()
    champion = data["champion"]["model_version"]
    challenger = data["challenger"]["model_version"]

    # Step 2: Evaluate accuracy
    winner = evaluate_models(champion, challenger)
    print(f"winner: {winner}")

    # Step 3: Decide next model versions and services
    current_version, service_to_update, new_model_version = update_model_versions(
        blob, data, winner, challenger
    )

    # Step 4: Trigger pipeline with resolved values
    job = trigger_pipeline(
        service_to_update=service_to_update,
        new_model_version=new_model_version,
        previous_model_version=current_version,
        use_caching=use_caching,
    )

    return {
        "message": "Pipeline job submitted successfully",
        "job_name": job.display_name,
        "prev_model_version": current_version,
        "new_model_version": new_model_version,
        "state": job.state.name,
        "caching_enabled": use_caching,
        "pipeline_url": f"https://console.cloud.google.com/vertex-ai/locations/{REGION}/pipelines/runs/{job.resource_name.split('/')[-1]}?project={PROJECT_ID}",
    }, 200
