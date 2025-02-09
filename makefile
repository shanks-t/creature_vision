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
	docker-compose -f docker/$(service)/docker-compose.local.yaml build --build-arg VERSION=$(VERSION)
	docker-compose -f docker/$(service)/docker-compose.local.yaml up

create_bq_table_tuning:
	bq mk \
  --table \
  --description="Table to store validation metrics for new model tuning" \
  creature-vision:dog_prediction_app.tuning_metrics \
  timestamp:TIMESTAMP,model_version:STRING,val_accuracy:FLOAT,loss:FLOAT,val_loss:FLOAT,training_duration:FLOAT,confusion_matrix:STRING,precision:FLOAT,recall:FLOAT,f1_score:FLOAT,class_distribution:STRING,learning_rate:FLOAT,hyperparams:STRING

create_bq_dataset:
	bq mk --dataset \
	--description="Dataset for creature vision project" \
	--location=US \
	creature-vision:dog_prediction_app

build_docker_cloud:
	VERSION=$(VERSION) docker-compose -f docker/$(service)/docker-compose.cloud.yaml build

auth_to_registry:
	gcloud auth configure-docker us-east1-docker.pkg.dev

push_to_registry:
	docker tag ${PROJECT_ID}/${APP_NAME}-$(service):${VERSION} ${IMAGE_TAG}
	docker push ${IMAGE_TAG}

cloud_run_deploy:
	gcloud run deploy dog-prediction-app --image=${IMAGE_TAG} --max-instances=1 --min-instances=0 --port=8080 \
 	--region=us-east1 --memory=2Gi --allow-unauthenticated -q

deploy_model:
	# make copy_model_from_gcs VERSION=$(VERSION)
	make build_docker_cloud VERSION=$(VERSION)
	make auth_to_registry
	make push_to_artifact_registry VERSION=$(VERSION)
	make cloud_run_deploy VERSION=$(VERSION)

run_local:
	make run_docker_local VERSION=$(VERSION)

	gcr.io/creature-vision/dog-prediction-app

	us-east1-docker.pkg.dev/creature-vision/dog-prediction-app
	
run_monitoring:
	docker compose -f docker-compose.grafana.yaml up