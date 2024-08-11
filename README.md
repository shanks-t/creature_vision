# Continual Learning for Animal Classification

## Project Overview

This project implements a continual learning system for animal classification using computer vision. The system will continuously improve its ability to recognize animals as new data comes in from user feedback and external sources.

## Technology Stack

- Python: Main programming language
- TensorFlow or PyTorch: For the computer vision model
- Docker: For containerization
- Terraform: For infrastructure as code
- Google Cloud Platform (GCP): Cloud provider
- Cloud Run: For serverless container deployment
- Cloud Storage: For data storage
- Cloud Functions: For serverless compute
- BigQuery: For data warehousing and analysis
- Vertex AI: For model training and deployment
- GitHub: For version control
- GitHub Actions: For CI/CD pipeline

## Implementation Steps

### 1. Setup and Initial Development

- Set up GCP project and enable necessary APIs
- Create a GitHub repository for the project
- Select and implement a pre-trained model for animal classification (e.g., MobileNetV2 pre-trained on ImageNet)
- Create Dockerfile for the model serving process

### 2. Infrastructure Setup

- Write Terraform scripts to set up GCP resources
- Set up GitHub Actions for CI/CD:
  - Lint and test Python code
  - Build and push Docker images to GCP Container Registry
  - Apply Terraform changes
  - Deploy to Cloud Run

### 3. Data Pipeline Implementation

- Develop Cloud Function for image acquisition:
  - Create an endpoint for user image uploads
  - Integrate with Unsplash or Pexels API for additional images
- Implement feedback collection system with predefined label set:
  - Create an endpoint that accepts an image and returns a prediction
  - Collect user feedback (correct/incorrect prediction)
  - If incorrect, allow user to select from predefined labels
  - Log each interaction in BigQuery

### 4. Continual Learning Loop Implementation

- Develop Cloud Function for retraining triggers:
  - Query BigQuery to check if retraining conditions are met
  - If conditions met, trigger retraining process
- Implement retraining pipeline using Vertex AI:
  - Create a custom training job that:
    - Loads the current model
    - Fetches new training data from Cloud Storage
    - Fine-tunes the model on new data
    - Saves the updated model
- Develop model evaluation logic:
  - Use a held-out test set to evaluate the new model
  - Compare performance against the current production model
- Create Cloud Function for model deployment:
  - If new model performs better, update the Cloud Run service with the new model
  - Log model update in BigQuery for tracking

### 5. Monitoring and Analysis Setup

- Configure Cloud Monitoring:
  - Set up alerts for model performance degradation
  - Monitor system health (e.g., Cloud Function executions, Cloud Run metrics)
- Create BigQuery queries for performance analysis:
  - Model accuracy over time
  - Distribution of animal classes in the dataset
  - User feedback trends
- Develop a dashboard using Data Studio or Looker:
  - Visualize key metrics and trends

### 6. Testing and Optimization

- Develop unit tests for each component of the pipeline
- Create integration tests to verify the end-to-end process
- Implement a staging environment for testing new models before production deployment
- Optimize Cloud Function performance and cost:
  - Adjust memory allocation and timeout settings
  - Implement caching where appropriate
- Fine-tune the sampling strategy in the data curation step

### 7. Documentation and Presentation

- Write comprehensive README.md in the GitHub repository:
  - Project overview
  - Architecture diagram
  - Setup instructions
  - Description of each component
- Create a blog post or video presentation:
  - Explain the continual learning concept
  - Showcase the MLOps practices implemented
  - Discuss challenges and learnings from the project

### 8. GitHub Actions CI/CD Pipeline

- Create GitHub Actions workflows:
  - On push to main branch:
    - Run linting and unit tests
    - Build Docker image and push to Container Registry
    - Apply Terraform changes (with approval step)
    - Deploy new version to Cloud Run
  - On pull request:
    - Run linting and unit tests
    - Perform Terraform plan
- Implement secrets management for GCP credentials

### 9. Data Ingestion and Retraining Automation

- Enhance the data curation process:
  - Implement a scoring mechanism for each new sample
  - Prioritize diverse and challenging samples for retraining
- Implement versioning for datasets:
  - Store each version of the dataset in a separate Cloud Storage folder
  - Keep metadata about each dataset version in BigQuery
- Enhance the retraining trigger function:
  - Consider model drift by comparing recent performance against historical metrics
  - Trigger retraining based on predefined conditions (e.g., performance threshold, time interval, data volume)

## Predefined Label Set

```python
VALID_LABELS = [
    "dog", "cat", "horse", "elephant", "giraffe", "lion", "tiger", "bear", 
    "zebra", "monkey", "bird", "fish", "rabbit", "deer", "snake", "other"
]
```

## Instructions
```
conda create -n creature_vis python=3.10 -c conda-forge
conda activate creature_vis 
conda install tensorflow pillow matplotlib numpy requests ipykernel -c conda-forge
pip install tensorflow-hub
```
