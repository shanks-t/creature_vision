# Use Google's pre-built TensorFlow CPU image for Vertex AI
FROM gcr.io/deeplearning-platform-release/tf2-cpu:latest

# Set working directory inside the container
WORKDIR /app

# Copy your training script and necessary files into the container
COPY . .

# Run the training script
CMD ["python", "-m", "main"]
