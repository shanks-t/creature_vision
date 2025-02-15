## TODO

- ~~containerize inference app~~
- ~~test locally with loading model from gcs~~
- decide where to run training service
- how to autmatically update the inference service with new model
- does the fine-tuned model need to be created as 'trainable' for the next tuning run?

### Ideas
- Convert the images to TFRecords and store them in a Cloud Storage bucket. Read the TFRecords by using the tf.data.TFRecordDataset function.
- Add a component to the Vertex AI pipeline that logs metrics to Vertex ML Metadata. Use Vertex AI Experiments to compare different executions of the pipeline. Use Vertex AI TensorBoard to visualize metrics.
- enabling caching for vertex ai preprocessing?
- Vertex ML Metadata is designed for ML metadata management, enabling efficient tracking of experiments, model parameters, and relationships between artifacts, making it an ideal choice for parameters storage (can be used with tensorboard logging)
- should I leverage vertexAI built-in container images?

- should I apply explainability metrics to model:
using Vertex Explainable AI to generate feature attributions. Aggregate feature attributions over the entire dataset. analyze the aggregation result together with the standard model evaluation metrics.

- can use dataflow for evalution using -runner=DataflowRunner in beam_pipeline_args

- Configure example-based explanations. Specify the embedding output layer to be used for the latent space representation. Example-based explanations analyze model behavior by comparing a given input to examples from the dataset, which can reveal patterns in the mislabeled images. By specifying the embedding output layer for the latent space representation, you can visualize and understand how the model groups images based on learned features. This approach helps identify clusters of similar images, revealing potential biases or anomalies that contribute to high-confidence misclassifications.

- Use Vertex Explainable AI to generate feature attributions. Aggregate feature attributions over the entire dataset. Analyze the aggregation result together with the standard model evaluation metrics. Aggregating feature attributions over the entire dataset allows you to identify patterns, such as which features consistently have a strong influence on the model's predictions. This aggregated view can help uncover potential biases or systemic issues in how the model processes different types of input data.

- I can Chain the Vertex AI ModelUploadOp and ModelDeployOp components together to configure a pipeline to upload a new version of the a model to Vertex AI Model Registry and deploy it to Vertex AI Endpoints for online inference 