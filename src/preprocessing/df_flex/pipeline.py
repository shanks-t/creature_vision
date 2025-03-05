from apache_beam.options.pipeline_options import PipelineOptions
import apache_beam as beam
import tensorflow as tf
from google.cloud import storage
import json
from datetime import datetime
from df_flex.logging_utils import setup_logger, timer, log_execution_time
import io
from PIL import Image
import numpy as np

logger = setup_logger('pipeline')


# @log_execution_time(logger)
def process_label_map(bucket_name: str, new_labels: set) -> dict:
    """Load existing label map or create new one, update if needed"""
    client = storage.Client(project='creature-vision')
    bucket = client.bucket(bucket_name)
    blob = bucket.blob('processed/metadata/label_map.json')

    with timer(logger, 'Loading label map'):
        if blob.exists():
            label_map = json.loads(blob.download_as_string())
            logger.info(
                f"Loaded existing label map with {len(label_map)} labels")
        else:
            label_map = {}
            logger.info("Created new label map")

        # Track if we need to update the file
        modified = False

        # Update map with new labels
        for label in new_labels:
            if label not in label_map:
                label_map[label] = len(label_map)
                modified = True
                logger.info(f"Added new label: {label}")

        # Only write back if modified
        if modified:
            blob.upload_from_string(json.dumps(label_map, indent=2))
            logger.info("Updated label map in GCS")

    return label_map


class GCSImagePathProvider(beam.PTransform):
    """Provides image paths from GCS for streaming processing with randomization"""

    def __init__(self, bucket_name, max_files=None, random_seed=None):
        super().__init__()
        self.bucket_name = bucket_name
        self.max_files = max_files
        self.random_seed = random_seed

    def expand(self, pcoll):
        def list_image_paths():
            import random

            # Set random seed if provided for reproducibility
            if self.random_seed is not None:
                random.seed(self.random_seed)

            client = storage.Client(project='creature-vision')
            bucket = client.bucket(self.bucket_name)

            # Collect all paths first
            all_paths = []
            for prefix in ['correct_predictions/', 'incorrect_predictions/']:
                blobs = bucket.list_blobs(
                    prefix=prefix,
                    fields='items(name)',
                    page_size=1000
                )

                for blob in blobs:
                    if blob.name.endswith('.jpg'):
                        all_paths.append(blob.name)

            # Shuffle all paths
            random.shuffle(all_paths)

            # Yield paths up to max_files limit
            count = 0
            for path in all_paths:
                yield path
                count += 1
                if self.max_files and count >= self.max_files:
                    return

        return (
            pcoll
            | 'Create' >> beam.Create([None])
            | 'ListImagePaths' >> beam.FlatMap(lambda _: list_image_paths())
        )


class ProcessImageAndLabel(beam.DoFn):
    """Process image and label from GCS paths"""

    def __init__(self, bucket_name, label_map):
        self.bucket_name = bucket_name
        self.label_map = label_map
        self.client = None
        self.bucket = None

    def setup(self):
        # Initialize GCS client during worker setup (runs once per worker)
        self.client = storage.Client(project='creature-vision')
        self.bucket = self.client.bucket(self.bucket_name)

    def process(self, image_path):
        try:
            # Derive label path from image path
            label_path = image_path.replace('.jpg', '_labels.json')

            # Download image
            image_blob = self.bucket.blob(image_path)
            image_bytes = image_blob.download_as_bytes()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_array = np.array(image)

            # Download label
            label_blob = self.bucket.blob(label_path)
            label_json = label_blob.download_as_text()
            label = json.loads(label_json)

            # Get label ID from map
            label_id = self.label_map.get(label['api_label'])
            if label_id is None:
                # Skip images with unknown labels
                return

            # Serialize image tensor
            image_bytes = tf.io.serialize_tensor(image_array).numpy()

            # Create and yield TFRecord example
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_id]))
            }
            example = tf.train.Example(features=tf.train.Features(
                feature=feature)).SerializeToString()

            yield example

        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")


class ProcessDataPipeline:
    def __init__(self, dataset_bucket_name: str, dataflow_bucket_name: str, batch_size: int = 32, project_id: str = 'creature-vision'):
        self.dataset_bucket_name = dataset_bucket_name
        self.dataflow_bucket_name = dataflow_bucket_name
        self.batch_size = batch_size
        self.project_id = project_id
        self.logger = setup_logger('ProcessDataPipeline')

    def _collect_labels(self, max_files=1):
        """Collect a sample of labels to build the label map"""
        client = storage.Client(project='creature-vision')
        bucket = client.bucket(self.dataset_bucket_name)

        labels = set()
        count = 0

        for prefix in ['correct_predictions/', 'incorrect_predictions/']:
            blobs = bucket.list_blobs(
                prefix=prefix,
                fields='items(name)',
                page_size=100
            )

            # Split between both directories
            for blob in list(blobs)[:max_files//2]:
                if blob.name.endswith('_labels.json'):
                    try:
                        label_json = blob.download_as_text()
                        label = json.loads(label_json)
                        labels.add(label['api_label'])
                        count += 1
                        if count >= max_files:
                            break
                    except Exception as e:
                        self.logger.error(
                            f"Error loading {blob.name}: {str(e)}")

        return labels

    @log_execution_time(logger)
    def run_pipeline(self, use_dataflow: bool = True, region: str = 'us-east1', max_files: int = None, random_seed: int = None,
                     max_num_workers: int = 2, number_of_worker_harness_threads: int = 4, machine_type: str = 'n1-standard-2'):
        """Process images and create TFRecords using Dataflow with streaming approach"""
        self.logger.info("Starting streaming pipeline")

        weekly_folder = f'weekly_{datetime.now().strftime("%Y%m%d")}'
        output_path = f'gs://{self.dataset_bucket_name}/processed/{weekly_folder}/data'
        self.logger.info(f"Output path: {output_path}")

        # Configure pipeline options
        pipeline_options = {
            'project': self.project_id,
            'temp_location': f'gs://{self.dataflow_bucket_name}/temp',
            'staging_location': f'gs://{self.dataflow_bucket_name}/staging',
        }

        # Add Dataflow-specific options if using Dataflow
        if use_dataflow:
            pipeline_options.update({
                'runner': 'DataflowRunner',
                'region': region,
                'job_name': f'creature-vision-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                'max_num_workers': max_num_workers,
                'disk_size_gb': 30,
                'experiments': ['use_runner_v2', 'use_preemptible_workers', 'flexrs_goal=COST_OPTIMIZED'],
                'machine_type': machine_type,
                'number_of_worker_harness_threads': number_of_worker_harness_threads,
                'autoscaling_algorithm': 'THROUGHPUT_BASED',
                'worker_disk_type': 'pd-standard',  # Standard disk is cheaper than SSD
                'save_main_session': True,  # Important for dependencies
            })
        else:
            pipeline_options['runner'] = 'DirectRunner'

        options = beam.options.pipeline_options.PipelineOptions(
            **pipeline_options)

        # First, collect a sample of labels to build the label map
        with timer(self.logger, 'Collecting labels for label map'):
            # Sample 1000 files to build label map
            sample_labels = self._collect_labels(max_files=1)
            self.logger.info(f"Collected {len(sample_labels)} unique labels")

            # Process label map once for all images
            label_map = process_label_map(
                self.dataset_bucket_name, sample_labels)

        self.logger.info("Starting Apache Beam pipeline")
        with timer(self.logger, 'Apache Beam pipeline execution'):
            with beam.Pipeline(options=options) as p:
                (p
                    | 'GetImagePaths' >> GCSImagePathProvider(self.dataset_bucket_name, max_files, random_seed=random_seed)
                    | 'ProcessImagesAndLabels' >> beam.ParDo(ProcessImageAndLabel(self.dataset_bucket_name, label_map))
                    | 'WriteTFRecord' >> beam.io.tfrecordio.WriteToTFRecord(
                        output_path,
                        file_name_suffix='.tfrecord',
                        num_shards=10  # Adjust based on your dataset size
                    ))

        self.logger.info("Pipeline completed successfully")
