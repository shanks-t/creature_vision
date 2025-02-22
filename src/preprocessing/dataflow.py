import apache_beam as beam
import tensorflow as tf
from google.cloud import storage
import json
from .data_loader import GCSDataLoader
from datetime import datetime
from .logging_utils import setup_logger, timer, log_execution_time


logger = setup_logger('pipeline')


@log_execution_time(logger)
def process_label_map(bucket_name: str, new_labels: str) -> dict:
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


@log_execution_time(logger)
def create_tfrecord_example(image: bytes, label_id: int) -> bytes:
    """Create single TFRecord example"""
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_id]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


class ProcessDataPipeline:
    def __init__(self, bucket_name: str, batch_size: int = 32):
        self.bucket_name = bucket_name
        self.batch_size = batch_size
        self.data_loader = GCSDataLoader(bucket_name)
        self.logger = setup_logger('ProcessDataPipeline')

    @log_execution_time(logger)
    def run_pipeline(self):
        """Process images and create TFRecords"""
        self.logger.info(
            f"Starting pipeline with batch size {self.batch_size}")

        options = beam.options.pipeline_options.PipelineOptions(
            project='creature-vision',
            temp_location=f'gs://{self.bucket_name}/temp'
        )

        weekly_folder = f'weekly_{datetime.now().strftime("%Y%m%d")}'
        output_path = f'gs://{self.bucket_name}/processed/{weekly_folder}/data'
        self.logger.info(f"Output path: {output_path}")

        with timer(self.logger, 'Loading raw batch'):
            images, labels = self.data_loader._load_raw_batch(self.batch_size)
            self.logger.info(
                f"Loaded {len(images)} images and {len(labels)} labels")
            # Get all unique labels from this batch
            new_labels = {label['api_label'] for label in labels}

            # Process label map once for all images
            label_map = process_label_map(self.bucket_name, new_labels)

        with timer(self.logger, 'Processing images and creating TFRecords'):
            # Create TFRecords
            examples = []
            for img, lbl in zip(images, labels):
                label_id = label_map[lbl['api_label']]
                example = create_tfrecord_example(
                    tf.io.serialize_tensor(img).numpy(),
                    label_id
                )
                examples.append(example)

        self.logger.info("Starting Apache Beam pipeline")
        with timer(self.logger, 'Apache Beam pipeline execution'):
            with beam.Pipeline(options=options) as p:
                (p
                 | 'CreateExamples' >> beam.Create(examples)
                 | 'WriteTFRecord' >> beam.io.tfrecordio.WriteToTFRecord(
                     output_path,
                     file_name_suffix='.tfrecord'
                 ))

        self.logger.info("Pipeline completed successfully")
