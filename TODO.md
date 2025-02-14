## TODO

- ~~containerize inference app~~
- ~~test locally with loading model from gcs~~
- decide where to run training service
- how to autmatically update the inference service with new model
- does the fine-tuned model need to be created as 'trainable' for the next tuning run?

### Ideas
- Convert the images to TFRecords and store them in a Cloud Storage bucket. Read the TFRecords by using the tf.data.TFRecordDataset function.
- Add a component to the Vertex AI pipeline that logs metrics to Vertex ML Metadata. Use Vertex AI Experiments to compare different executions of the pipeline. Use Vertex AI TensorBoard to visualize metrics.
