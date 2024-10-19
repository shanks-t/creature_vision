# Continuous Retraining Project

This project demonstrates a 'data flywheel' effect using a simulated user interaction system with a machine learning model. It leverages various technologies to create a continuous learning and improvement cycle.

## Overview

The Continuous Retraining Project uses the free Dog API to simulate user interactions, feeding data into a containerized MobileNetV3 model running on Cloud Run. The system collects performance metrics and new training data, enabling continuous model improvement.

## Key Features

- **Simulated User Interactions**: Utilizes the free Dog API to generate realistic user interaction data.
- **Containerized ML Model**: Runs a lightweight MobileNetV3 model in a container on Cloud Run for efficient and scalable predictions.
- **Performance Tracking**: Outputs model performance metrics to BigQuery for detailed analysis.
- **Metric Visualization**: Integrates with Grafana using a BigQuery data source to visualize model performance over time.
- **Automated Data Collection**: Saves new training data (images and labels) from the Dog API for future model fine-tuning.
- **Threshold-based Retraining**: Initiates model fine-tuning once a specified volume of new training data is accumulated.

## Architecture

1. **Data Source**: [Free Dog API](https://dog.ceo/dog-api/about)
2. **Model Hosting**: Cloud Run (containerized MobileNetV3)
3. **Metric Storage**: BigQuery
4. **Visualization**: Grafana
5. **Data Storage**: Cloud Storage (for new training data)

## Workflow

1. The system fetches dog images and labels from the Dog API.
2. These images are sent to the MobileNetV3 model hosted on Cloud Run for prediction.
3. Prediction results and actual labels are compared to generate performance metrics.
4. Metrics are stored in BigQuery for analysis.
5. Grafana visualizes the performance metrics from BigQuery.
6. New images and labels are saved to Cloud Storage for future model fine-tuning.
7. When the volume of new data reaches a predefined threshold, the model is retrained to improve its performance.

## Getting Started

TODO: (Include instructions for setting up and running the project)

## Monitoring
- Once your gcp service account key is created, downloaded to your host machine, you can mount the key to you grafana container
- The dashboards and datasource configs are stored in grafana monitoring dir. mount these dirs as well, and the 'creature-vision' dashbpoard viz will automatically startup
![accuracy](./docs/grafana.png)


## Fine-tuning
- https://www.tensorflow.org/tutorials/images/transfer_learning