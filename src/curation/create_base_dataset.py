import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from google.cloud import storage

# Define the GCS path
gcs_path = "gs://creature-vision-training-set/v3_0"
bucket_name = "creature-vision-training-set"


class ListGCSJPGPaths(beam.PTransform):
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name

    def expand(self, pcoll):
        return pcoll | "List .jpgs in GCS" >> beam.ParDo(
            ListJPGsFromPrefixDoFn(self.bucket_name)
        )


class ListJPGsFromPrefixDoFn(beam.DoFn):
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name

    def setup(self):
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)

    def process(self, prefix):
        # prefix = "correct_predictions/" or "incorrect_predictions/"
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
        for blob in blobs:
            if blob.name.endswith(".jpg"):
                yield blob.name


# Create a pipeline
with beam.Pipeline(
    options=PipelineOptions(
        runner="DirectRunner",
        direct_running_mode="multi-threading",
        direct_num_workers=12,
    )
) as p:
    prefixes = p | "Prefixes" >> beam.Create(
        ["correct_predictions/", "incorrect_predictions/"]
    )

    jpg_paths = prefixes | "List JPGs" >> ListGCSJPGPaths(bucket_name=bucket_name)

    # Optional: just print results for debugging
    jpg_paths | "Print paths" >> beam.Map(print)
