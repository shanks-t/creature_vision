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

## Hybrid Approach

### Overview
- Use Google Cloud Monitoring for data collection
- Export aggregated data periodically to local InfluxDB instance
- Visualize data using locally hosted Grafana

### Benefits
1. Leverages GCP's built-in monitoring capabilities
2. Minimizes data transfer and storage costs by aggregating before export
3. Maintains local control over visualization and long-term data storage
4. Simplifies setup by using cloud services for data collection

### Implementation Steps

#### 1. Set up Google Cloud Monitoring
- Enable Cloud Monitoring API in your GCP project
- Configure custom metrics for:
  * Model accuracy
  * Number of images processed
  * Number of incorrect predictions
  * Retraining events
- Set up log-based metrics for specific events

#### 2. Create a Cloud Function for Data Export
- Develop a Python Cloud Function that:
  * Queries Cloud Monitoring for aggregated metrics
  * Formats data for InfluxDB
  * Securely sends data to your local InfluxDB instance
- Deploy the Cloud Function to GCP
- Set up Cloud Scheduler to trigger the function periodically (e.g., hourly)

#### 3. Configure Local VM
- Set up a VM on your local network (e.g., using VirtualBox or VMware)
- Install Ubuntu Server or another Linux distribution
- Configure network to allow incoming connections from GCP and local network

#### 4. Install and Configure InfluxDB
- Install InfluxDB on the local VM
- Create a database for storing monitoring data
- Set up authentication and secure the database
- Configure firewall to allow incoming connections from GCP

#### 5. Install and Configure Grafana
- Install Grafana on the local VM
- Configure Grafana to use InfluxDB as a data source
- Set up user authentication for Grafana

#### 6. Create Grafana Dashboards
- Design dashboards for:
  * Model performance overview
  * Data collection statistics
  * Retraining events and model versions
  * System health metrics

#### 7. Set up Secure Communication
- Configure VPN or SSH tunnel between GCP and local network
- Use HTTPS for all data transfers
- Implement proper authentication for the Cloud Function to access local InfluxDB

#### 8. Implement Alerting
- Set up Grafana alerting for critical metrics
- Configure email or messaging integration for alerts

### Maintenance and Optimization
- Regularly review and optimize Cloud Monitoring metrics
- Adjust data aggregation and export frequency based on needs
- Periodically update Grafana dashboards for improved visualizations
- Monitor local VM performance and scale resources as needed

### Considerations
- **Data Retention**: Determine how long to keep detailed data in InfluxDB
- **Backup Strategy**: Implement regular backups of InfluxDB data
- **Network Security**: Ensure proper firewall and access controls between GCP and local network
- **Scalability**: Plan for increased data volume as the project grows

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
