# Standard Library Imports
import os
import io
import json
from datetime import datetime
import random

# Third-Party Imports
import apache_beam as beam
import tensorflow as tf
import numpy as np
from PIL import Image
from google.cloud import storage

# Local/Project-Specific Imports
from df_flex.logging_utils import setup_logger, timer, log_execution_time

logger = setup_logger("pipeline")


class FormatLabelStatsForBigQuery(beam.DoFn):
    def __init__(self, model_version: str):
        self.model_version = model_version

    def process(self, element, timestamp=beam.DoFn.TimestampParam):
        """
        element: Tuple[((class_name: str, label_id: int), count: int)]
        """
        (class_name, label_id), example_count = element

        # Handle undefined or invalid timestamp
        try:
            ts_str = timestamp.to_rfc3339()
        except OverflowError:
            ts_str = datetime.utcnow().isoformat() + "Z"

        yield {
            "model_version": self.model_version,
            "class_label": class_name,
            "label_id": label_id,
            "example_count": example_count,
            "run_timestamp": ts_str,
        }


class GCSImagePathProvider(beam.PTransform):
    """Provides image paths from GCS for streaming processing"""

    def __init__(self, bucket_name, max_files=None, random_seed=None):
        super().__init__()
        self.bucket_name = bucket_name
        self.max_files = max_files
        self.random_seed = random_seed

    def expand(self, pcoll):
        def list_image_paths():
            if self.random_seed is not None:
                random.seed(self.random_seed)

            client = storage.Client(project="creature-vision")
            bucket = client.bucket(self.bucket_name)

            all_paths = []
            for prefix in ["incorrect_predictions/"]:
                # Correct method to list blobs
                blobs = bucket.list_blobs(prefix=prefix)
                for blob in blobs:
                    if blob.name.endswith(".jpg"):
                        all_paths.append(blob.name)

            random.shuffle(all_paths)
            return all_paths[: self.max_files] if self.max_files else all_paths

        return pcoll | beam.Create(list_image_paths())


class ProcessImageAndLabel(beam.DoFn):
    """Processes images and their labels from GCS"""

    def __init__(self, bucket_name, label_map):
        self.bucket_name = bucket_name
        self.label_map = label_map
        self.client = None
        self.bucket = None

    def setup(self):
        self.client = storage.Client(project="creature-vision")
        self.bucket = self.client.bucket(self.bucket_name)

    def process(self, image_path):
        # Apache Beam expects the process method to:
        # Return an iterable (list or generator).
        outputs = []

        try:
            label_path = image_path.replace(".jpg", "_labels.json")
            filename = os.path.basename(image_path).split(".")[0]

            # Fetch the image and label blobs individually
            image_blob = self.bucket.blob(image_path)
            label_blob = self.bucket.blob(label_path)

            # Check if the label blob exists, and if not, skip this entry
            if not label_blob.exists():
                return  # Label missing, skip this image

            image_bytes = image_blob.download_as_bytes()
            label_json = label_blob.download_as_text()
            label = json.loads(label_json)
            class_name = label["api_label"]
            label_id = self.label_map.get(class_name)

            if label_id is None:
                # Returning [] explicitly tells Beam this function ran successfully but produced no output.
                # If None is returned, Beam ignores it—no error, no warning.
                # This leads to silent data loss, as those elements simply disappear from the pipeline.
                return

            image_array = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
            image_bytes = tf.io.serialize_tensor(image_array).numpy()

            feature = {
                "image": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image_bytes])
                ),
                "label": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label_id])
                ),
                "filename": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[filename.encode()])
                ),
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=feature)
            ).SerializeToString()

            # Yield TFRecord to main output
            yield beam.pvalue.TaggedOutput("tfrecord", example)

            # Yield class_name and label_id to stats output
            yield beam.pvalue.TaggedOutput("label_stats", (class_name, label_id))

        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")

        return  # Return nothing


class ProcessDataPipeline:
    def __init__(
        self,
        dataset_bucket_name: str,
        dataflow_bucket_name: str,
        batch_size: int = 32,
        project_id: str = "creature-vision",
    ):
        self.dataset_bucket_name = dataset_bucket_name
        self.dataflow_bucket_name = dataflow_bucket_name
        self.batch_size = batch_size
        self.project_id = project_id
        self.logger = setup_logger("ProcessDataPipeline")

    @log_execution_time(logger)
    def run_pipeline(
        self,
        version: str,
        use_dataflow: bool = True,
        region: str = "us-east1",
        max_files: int = None,
        random_seed: int = None,
        max_num_workers: int = 2,
        number_of_worker_harness_threads: int = 4,
        machine_type: str = "n1-standard-2",
    ):
        """Runs the Apache Beam pipeline to process images and create TFRecords"""

        self.logger.info("Updating label map before starting the pipeline...")

        # **Step 2**: Load the updated label map
        client = storage.Client(project="creature-vision")
        bucket = client.bucket(self.dataset_bucket_name)
        blob = bucket.blob("processed/metadata/label_map.json")
        label_map = json.loads(blob.download_as_text())

        output_path = f"gs://{self.dataset_bucket_name}/processed/{version}/data"
        self.logger.info(f"Output path: {output_path}")

        # **Step 3**: Configure Beam pipeline options
        pipeline_options = {
            "project": self.project_id,
            "temp_location": f"gs://{self.dataflow_bucket_name}/temp",
            "staging_location": f"gs://{self.dataflow_bucket_name}/staging",
        }

        if use_dataflow:
            pipeline_options.update(
                {
                    "runner": "DataflowRunner",
                    "region": region,
                    "job_name": f"creature-vision-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    "max_num_workers": max_num_workers,
                    "disk_size_gb": 30,
                    "experiments": [
                        "use_runner_v2",
                        "use_preemptible_workers",
                        "flexrs_goal=COST_OPTIMIZED",
                    ],
                    "machine_type": machine_type,
                    "number_of_worker_harness_threads": number_of_worker_harness_threads,
                    "autoscaling_algorithm": "THROUGHPUT_BASED",
                    "worker_disk_type": "pd-standard",
                    "save_main_session": True,
                }
            )
        else:
            pipeline_options["runner"] = "DirectRunner"

        options = beam.options.pipeline_options.PipelineOptions(**pipeline_options)

        # **Step 4**: Start Beam pipeline
        self.logger.info("Starting Apache Beam pipeline")
        with timer(self.logger, "Apache Beam pipeline execution"):
            with beam.Pipeline(options=options) as p:
                results = (
                    p
                    | "GetImagePaths"
                    >> GCSImagePathProvider(
                        self.dataset_bucket_name, max_files, random_seed=random_seed
                    )
                    | "ProcessImagesAndLabels"
                    >> beam.ParDo(
                        ProcessImageAndLabel(self.dataset_bucket_name, label_map)
                    ).with_outputs("tfrecord", "label_stats")
                )

                # TFRecord output (unchanged)
                _ = (
                    results.tfrecord
                    | "WriteTFRecord"
                    >> beam.io.tfrecordio.WriteToTFRecord(
                        output_path,
                        file_name_suffix=".tfrecord",
                    )
                )

                # NEW: Class distribution stats to BigQuery
                (
                    results.label_stats
                    | "CountByClass" >> beam.combiners.Count.PerElement()
                    | "FormatBQRow"
                    >> beam.ParDo(FormatLabelStatsForBigQuery(model_version=version))
                    | "WriteStatsToBQ"
                    >> beam.io.WriteToBigQuery(
                        table=f"{self.project_id}.training_metrics.class_distribution",
                        schema={
                            "fields": [
                                {
                                    "name": "model_version",
                                    "type": "STRING",
                                    "mode": "REQUIRED",
                                },
                                {
                                    "name": "class_label",
                                    "type": "STRING",
                                    "mode": "REQUIRED",
                                },
                                {
                                    "name": "label_id",
                                    "type": "INTEGER",
                                    "mode": "REQUIRED",
                                },
                                {
                                    "name": "example_count",
                                    "type": "INTEGER",
                                    "mode": "REQUIRED",
                                },
                                {
                                    "name": "run_timestamp",
                                    "type": "TIMESTAMP",
                                    "mode": "REQUIRED",
                                },
                            ]
                        },
                        write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                    )
                )

        self.logger.info("Pipeline completed successfully")
