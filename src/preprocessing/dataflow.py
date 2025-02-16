import apache_beam as beam
import tensorflow as tf
from google.cloud import storage
import json
from data_loader import GCSDataLoader
from datetime import datetime


def load_or_update_label_map(bucket_name: str, new_label: str) -> dict:
    """Load existing label map or create new one, update if needed"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob('processed/metadata/label_map.json')

    # Load existing map or create new
    if blob.exists():
        label_map = json.loads(blob.download_as_string())
    else:
        label_map = {}

    # Update map if new label found
    if new_label not in label_map:
        label_map[new_label] = len(label_map)
        blob.upload_from_string(json.dumps(label_map, indent=2))

    return label_map


def create_tfrecord_example(image: bytes, label: str, label_id: int) -> bytes:
    """Create single TFRecord example"""
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_id])),
        'label_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode()]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


class ProcessDataPipeline:
    def __init__(self, bucket_name: str, batch_size: int = 32):
        self.bucket_name = bucket_name
        self.batch_size = batch_size
        self.data_loader = GCSDataLoader(bucket_name)

    def run_pipeline(self):
        """Process images and create TFRecords"""
        # Setup pipeline options
        options = beam.options.pipeline_options.PipelineOptions(
            project='creature-vision',
            temp_location=f'gs://{self.bucket_name}/temp'
        )

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'gs://{self.bucket_name}/processed/data_{timestamp}'

        with beam.Pipeline(options=options) as p:
            # Load raw batch
            images, labels = self.data_loader._load_raw_batch(self.batch_size)

            examples = []
            for img, lbl in zip(images, labels):
                # Get or create label mapping
                _, label_id = load_or_update_label_map(
                    self.bucket_name,
                    lbl['api_label']
                )

                # Create TFRecord with integer label
                example = create_tfrecord_example(
                    tf.io.serialize_tensor(img).numpy(),
                    label_id
                )
                examples.append(example)

            # Write to TFRecord
            (p
             | 'CreateExamples' >> beam.Create(examples)
             | 'WriteTFRecord' >> beam.io.tfrecordio.WriteToTFRecord(
                 output_path,
                 file_name_suffix='.tfrecord'
             ))


# Usage
pipeline = ProcessDataPipeline('creature-vision-training-set', batch_size=10)
pipeline.run_pipeline()
