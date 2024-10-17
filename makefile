APP_NAME := dog-prediction-app
PROJECT_ID := creature-vision
HOSTNAME := gcr.io
REGION := us-east1
IMAGE_NAME := c-vis
ARTIFACT_REGISTRY := ${REGION}-docker.pkg.dev
IMAGE_TAG := ${ARTIFACT_REGISTRY}/${PROJECT_ID}/${APP_NAME}/${VERSION}

build_docker:
	docker build . -t ${APP_NAME}:latest -t ${APP_NAME}:${VERSION}

run_docker_local:
	docker-compose -f docker-compose.local.yaml build --build-arg VERSION=$(VERSION)
	docker-compose -f docker-compose.local.yaml up

create_bq_table:
	bq mk \
  --table \
  --description="Table to store prediction metrics for the dog breed classifier" \
  creature-vision:dog_prediction_app.prediction_metrics \
  timestamp:TIMESTAMP,actual:STRING,predicted:STRING,is_correct:BOOLEAN,latency:FLOAT

create_bq_dataset:
	bq mk --dataset \
	--description="Dataset for creature vision project" \
	--location=US \
	creature-vision:dog_prediction_app

copy_to_gcs: $(model)
	gsutil -m cp $(model) gs://tf_models_cv/$(model)

copy_model_from_gcs:
	gsutil cp gs://tf_models_cv/m_net_$(VERSION).keras ./model.keras

build_docker_cloud:
	VERSION=$(VERSION) docker-compose -f docker-compose.cloud.yaml build

auth_to_registry:
	gcloud auth configure-docker us-east1-docker.pkg.dev

push_to_registry:
	docker tag ${PROJECT_ID}/${APP_NAME}:${VERSION} ${IMAGE_TAG}
	docker push ${IMAGE_TAG}

cloud_run_deploy:
	gcloud run deploy dog-prediction-app --image=${IMAGE_TAG} --max-instances=2 --min-instances=0 --port=8080 \
 	--region=us-east1 --memory=2Gi -q

deploy_model:
	# make copy_model_from_gcs VERSION=$(VERSION)
	make build_docker_cloud VERSION=$(VERSION)
	make auth_to_registry
	make push_to_artifact_registry VERSION=$(VERSION)
	make cloud_run_deploy VERSION=$(VERSION)

run_local:
	make copy_model_from_gcs VERSION=$(VERSION)
	make run_docker_local VERSION=$(VERSION)

	gcr.io/creature-vision/dog-prediction-app

	us-east1-docker.pkg.dev/creature-vision/dog-prediction-app
	
run_monitoring:
	docker compose -f docker-compose.grafana.yaml up