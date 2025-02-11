## TODO

- ~~containerize inference app~~
- ~~test locally with loading model from gcs~~
- decide where to run training service
- how to autmatically update the inference service with new model
- does the fine-tuned model need to be created as 'trainable' for the next tuning run?

### Ideas
- Convert the images to TFRecords and store them in a Cloud Storage bucket. Read the TFRecords by using the tf.data.TFRecordDataset function.

