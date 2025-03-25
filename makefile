# Project Configuration
PROJECT_ID := creature-vision
REGION := us-east1
VERSION ?= latest

# Registry Configuration
ARTIFACT_REGISTRY := ${REGION}-docker.pkg.dev

# Service Names
INFERENCE_APP := dog-prediction-app
TRAINING_APP := creature-vis-training
PREPROCESSING_APP := creature-vis-preprocessing

# Validate Service Parameter
VALID_SERVICES := inference training preprocessing
SERVICE ?= inference
ifeq ($(filter $(SERVICE),$(VALID_SERVICES)),)
    $(error Invalid service. Must be one of: $(VALID_SERVICES))
endif

# Service-specific variables
ifeq ($(SERVICE),inference)
    APP_NAME := $(INFERENCE_APP)
else ifeq ($(SERVICE),preprocessing)
    APP_NAME := $(PREPROCESSING_APP)
else
    APP_NAME := $(TRAINING_APP)
endif

# Image Tag
IMAGE_TAG := ${ARTIFACT_REGISTRY}/${PROJECT_ID}/${APP_NAME}/${SERVICE}:${VERSION}
TAR_FILE=${SERVICE}.tar  # Define the tar file for saving the image
# Dataflow parameters
MAXFILES ?= 500
RANDOM_SEED ?= 42

# Template variables for test-template
GCP_PROJECT ?= $(PROJECT_ID)
GCP_REGION ?= $(REGION)
TEMPLATE_IMAGE ?= ${ARTIFACT_REGISTRY}/${PROJECT_ID}/${PREPROCESSING_APP}/preprocessing:${VERSION}

.PHONY: help build run-local build-cloud push-ar push-gcr auth-registry configure-kubectl get-deps test-template

help:
	@echo "Available commands:"
	@echo "  make build SERVICE=[inference|training|preprocessing]     - Build docker image locally"
	@echo "  make run-local SERVICE=[inference|training|preprocessing] - Run service locally"
	@echo "  make build-cloud SERVICE=[inference|training|preprocessing] - Build for cloud deployment"
	@echo "  make push-ar SERVICE=[inference|training|preprocessing]   - Push to Artifact Registry"
	@echo "  make push-gcr SERVICE=[inference|training|preprocessing]  - Push to Container Registry"
	@echo "  make auth-registry                         - Configure docker auth"
	@echo "  make configure-kubectl                     - Configure kubectl"
	@echo "  make get-deps SERVICE=[inference|training|preprocessing]  - Generate requirements.txt for service"
	@echo "  make test-template                         - Test the integrity of the Flex Container"

get-deps:
	@echo "Generating requirements.txt for $(SERVICE) service..."
	pip install pipreqs
	pipreqs ./src/$(SERVICE) --savepath ./src/$(SERVICE)/requirements.txt --use-local --force

build-push: auth-registry
	docker buildx build -f docker/$(SERVICE)/Dockerfile src/$(SERVICE)/ \
		--platform linux/amd64 \
		-t ${IMAGE_TAG} \
		--push

run-local:
	docker-compose -f docker/$(SERVICE)/docker-compose.local.yaml build \
		--build-arg VERSION=$(VERSION)
	docker-compose -f docker/$(SERVICE)/docker-compose.local.yaml up && \
	docker compose -f docker/$(SERVICE)/docker-compose.local.yaml rm -fsv

# Authenticate to Artifact Registry
auth-registry:
	gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build image without pushing (disable BuildKit to avoid timeout issues)
build:
	DOCKER_BUILDKIT=0 docker build -f docker/$(SERVICE)/Dockerfile src/$(SERVICE)/ \
		--platform linux/amd64 \
		-t ${IMAGE_TAG}


push-image: auth-registry
	docker tag ${APP_NAME}:${VERSION} ${IMAGE_TAG}
	docker push ${IMAGE_TAG}

create-df-template:
	gcloud dataflow flex-template build gs://dataflow-use1/templates/creature-vision-template.json \
	--image=us-east1-docker.pkg.dev/creature-vision/creature-vis-preprocessing/preprocessing:latest \
	--sdk-language=PYTHON \
	--metadata-file=./src/preprocessing/metadata.json

run-dataflow:
	@echo "Running Dataflow job with max_files=$(MAXFILES)..."
	gcloud dataflow flex-template run "creature-vis-processing" \
	--template-file-gcs-location=gs://dataflow-use1/templates/creature-vision-template.json \
	--region=us-east1 \
	--parameters=max_files=$(MAXFILES)

run-monitoring:
	docker compose -f docker-compose.grafana.yaml up

test-template: ## Test the Integrity of the Flex Container
	@gcloud config set project ${GCP_PROJECT}
	@gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev
	@docker pull --platform linux/amd64 ${TEMPLATE_IMAGE}
	@echo "Checking if ENV Var FLEX_TEMPLATE_PYTHON_PY_FILE is Available" && docker run --platform linux/amd64 --rm --entrypoint /bin/bash ${TEMPLATE_IMAGE} -c 'env|grep -q "FLEX_TEMPLATE_PYTHON_PY_FILE" && echo ✓'
	@echo "Checking if ENV Var FLEX_TEMPLATE_PYTHON_SETUP_FILE is Available" && docker run --platform linux/amd64 --rm --entrypoint /bin/bash ${TEMPLATE_IMAGE} -c 'env|grep -q "FLEX_TEMPLATE_PYTHON_PY_FILE" && echo ✓'
	@echo "Checking if Driver Python File (main.py) Found on Container" && docker run --platform linux/amd64 --rm --entrypoint /bin/bash ${TEMPLATE_IMAGE} -c "/usr/bin/test -f ${FLEX_TEMPLATE_PYTHON_PY_FILE} && echo ✓"
	@echo "Checking if setup.py File Found on Container" && docker run --platform linux/amd64 --rm --entrypoint /bin/bash ${TEMPLATE_IMAGE} -c 'test -f ${FLEX_TEMPLATE_PYTHON_SETUP_FILE} && echo ✓'
	@echo "Checking if Package Installed on Container" && docker run --platform linux/amd64 --rm --entrypoint /bin/bash ${TEMPLATE_IMAGE} -c 'python -c "import beam_flex" && echo ✓'
	@echo "Checking if UDFs Installed on Container" && docker run --platform linux/amd64 --rm --entrypoint /bin/bash ${TEMPLATE_IMAGE} -c 'python -c "from beam_flex.modules.pipeline import GCSImagePathProvider" && echo ✓'
	@echo "Running Pipeline Locally..." && docker run --platform linux/amd64 --rm --entrypoint /bin/bash ${TEMPLATE_IMAGE} -c "python ${FLEX_TEMPLATE_PYTHON_PY_FILE} --runner DirectRunner --output output.txt && cat output.txt*"

# Cloud Run Inference Test
test-inference:
	@echo "Sending test prediction request to Cloud Run inference service..."
	curl -X GET\
	  https://dog-predictor-284159624099.us-east1.run.app/predict \
	  
test-train: cp-train-pkg
	@echo "Submitting custom training job to test new training code"
	gcloud ai custom-jobs create \
	  --region=us-east1 \
	  --display-name=creture-vision-training \
	  --python-package-uris=gs://creture-vision-ml-artifacts/python_packages/creature_vision_training-0.1.tar.gz \
	  --args=--version=v-20250323 \
	  --worker-pool-spec=machine-type=e2-standard-4,replica-count=1,executor-image-uri=us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest,python-module=creature_vision_training.main \
	  --service-account=kubeflow-pipeline-sa@creature-vision.iam.gserviceaccount.com

cp-train-pkg:
	cd src/training && \
	rm -rf dist && \
	python setup.py sdist && \
	gsutil cp dist/*.tar.gz gs://creture-vision-ml-artifacts/python_packages/

check-pkg:
	gsutil cp gs://creture-vision-ml-artifacts/python_packages/creature_vision_training-0.1.tar.gz - | tar -tzf -

test-run-inf:
	curl -X GET https://dog-predictor-284159624099.us-east1.run.app/predict/

test-pipeline-cf:
	curl -X POST https://us-east1-creature-vision.cloudfunctions.net/trigger-creature-pipeline

deploy-pipeline-cf: compile-pipeline
	cd src/trigger && \
	gcloud functions deploy trigger-creature-pipeline \
	--runtime=python310 --entry-point=trigger_pipeline \
	--trigger-http --region=us-east1 --memory=512MB \
	--source=. --allow-unauthenticated

compile-pipeline:
	@echo "📦 Compiling Kubeflow pipeline..."
	python src/kubeflow/compile_pipeline.py
	@echo "🚀 Uploading pipeline JSON to GCS..."
	gsutil cp creature_vision_pipeline.json gs://creature-vision-pipeline-artifacts/kubeflow-templates/
	@echo "✅ Pipeline compilation and upload complete."
