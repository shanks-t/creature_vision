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

.PHONY: help build run-local build-cloud push-ar push-gcr auth-registry configure-kubectl get-deps

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

get-deps:
	@echo "Generating requirements.txt for $(SERVICE) service..."
	pip install pipreqs
	pipreqs ./src/$(SERVICE) --savepath ./src/$(SERVICE)/requirements.txt --use-local --force

build:
	docker build -f docker/$(SERVICE)/Dockerfile . \
		-t ${APP_NAME}:${VERSION} \
		-t ${APP_NAME}:latest

run-local:
	docker-compose -f docker/$(SERVICE)/docker-compose.local.yaml build \
		--build-arg VERSION=$(VERSION)
	docker-compose -f docker/$(SERVICE)/docker-compose.local.yaml up && \
	docker compose -f docker/$(SERVICE)/docker-compose.local.yaml rm -fsv

build-push: auth-registry
	docker buildx build -f docker/$(SERVICE)/Dockerfile . \
		--platform linux/amd64 \
		-t ${IMAGE_TAG} \
		--push

auth-registry:
	gcloud auth configure-docker ${REGION}-docker.pkg.dev

push-image: auth-registry
	docker tag ${APP_NAME}:${VERSION} ${IMAGE_TAG}
	docker push ${IMAGE_TAG}

configure-kubectl:
	gcloud container clusters get-credentials ml-cv-cluster \
		--zone us-east1-b \
		--project $(PROJECT_ID)

# Monitoring remains unchanged
run-monitoring:
	docker compose -f docker-compose.grafana.yaml up
