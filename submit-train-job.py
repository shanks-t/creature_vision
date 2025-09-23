from google.cloud import aiplatform

aiplatform.init(
    project="creature-vision",
    location="us-east1",
    staging_bucket="gs://creture-vision-ml-artifacts",
)

job = aiplatform.CustomPythonPackageTrainingJob(
    display_name="creature-vision-training",
    python_package_gcs_uri="gs://creture-vision-ml-artifacts/python_packages/creature_vision_training-0.1.tar.gz",
    python_module_name="creature_vision_training.main",
    container_uri="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest",
)

job.run(
    args=["--version=v3_2", "--previous_model_version=v3_1"],
    replica_count=1,
    machine_type="e2-standard-4",
    tensorboard="projects/284159624099/locations/us-east1/tensorboards/3018185806524186624",
    service_account="kubeflow-pipeline-sa@creature-vision.iam.gserviceaccount.com",
)
