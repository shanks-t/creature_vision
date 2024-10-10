APP_NAME := dog-prediction-app

build_docker:
	docker build . -t ${APP_NAME}:latest -t ${APP_NAME}:${VERSION}

run_docker_local:
	docker-compose -f docker-compose.local.yaml up --build

build_docker_cloud:
	docker-compose -f docker-compose.local.yaml --build

PROJECT_ID := $(shell gcloud config get-value project)
HOSTNAME := gcr.io
GCR_TAG := ${HOSTNAME}/${PROJECT_ID}/${APP_NAME}:${VERSION}

run_grc_build:
	echo "${GCR_TAG}"
	gcloud builds submit --tag ${GCR_TAG} -q

cloud_run_deploy:
	gcloud run deploy ner-app --image=${GCR_TAG} --max-instances=2 --min-instances=0 --port=8080 \
	--allow-unauthenticated --region=europe-west1 --memory=2Gi --cpu=4 -q

cloud_run_delete:
	gcloud run services delete ner-app --region=europe-west1 -q

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