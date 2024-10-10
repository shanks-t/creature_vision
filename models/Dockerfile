FROM tensorflow/serving:latest

# Install Google Cloud SDK
RUN apt-get update && apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-sdk

# Copy the model config file
COPY models.config /models/models.config

# Set the model config file path
ENV MODEL_CONFIG_FILE=/models/models.config

EXPOSE 8501

# Use the TensorFlow Serving entrypoint
ENTRYPOINT ["tensorflow_model_server", "--model_config_file=models/models.config", "--port=8500", "--rest_api_port=8501"]