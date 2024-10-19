import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.cloud import storage, bigquery
import os
import tempfile
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

storage_client = storage.Client(project="creature-vision")
bigquery_client = bigquery.Client(project="creature-vision")

data_bucket_name = "creature-vision-training-set"
data_bucket = storage_client.bucket(data_bucket_name)
model_bucket_name = "tf_models_cv"
model_bucket = storage_client.bucket(model_bucket_name)

def download_model_from_gcs(model_blob_name, local_path):
    logger.debug(f"Downloading model {model_blob_name} from GCS")
    blob = model_bucket.blob(model_blob_name)
    blob.download_to_filename(local_path)
    logger.debug(f"Model downloaded to {local_path}")

def download_data_from_gcs(prefix, local_dir):
    logger.debug(f"Downloading data with prefix {prefix} from GCS")
    blobs = data_bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        filename = os.path.basename(blob.name)
        local_path = os.path.join(local_dir, filename)
        blob.download_to_filename(local_path)
    logger.debug(f"Data downloaded to {local_dir}")

def fine_tune_model(data_dir, model_path, new_model_path):
    logger.debug("Starting model fine-tuning")
    model = tf.keras.models.load_model(model_path)
    
    for layer in model.layers[:-3]:
        layer.trainable = False
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    logger.debug("Starting model training")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 32,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // 32,
        epochs=10
    )
    
    logger.debug(f"Saving fine-tuned model to {new_model_path}")
    model.save(new_model_path)
    
    return history.history

def upload_model_to_gcs(source_file_name, destination_blob_name):
    logger.debug(f"Uploading model from {source_file_name} to GCS as {destination_blob_name}")
    blob = model_bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    logger.debug("Model upload complete")

def get_latest_model_version():
    logger.debug("Fetching latest model version from GCS")
    blobs = list(model_bucket.list_blobs(prefix='m_net_v'))
    versions = [float(blob.name.split('_v')[1].split('.keras')[0]) for blob in blobs]
    latest_version = max(versions) if versions else 0.0
    logger.debug(f"Latest model version: {latest_version}")
    return latest_version

def increment_version(version):
    major, minor = str(version).split('.')
    new_version = f"{major}.{int(minor) + 1}"
    logger.debug(f"Incrementing version from {version} to {new_version}")
    return new_version

def record_validation_data(version, history):
    logger.debug(f"Recording validation data for model version {version}")
    table_id = "creature-vision.dog_prediction_app.model_training_metrics"
    
    rows_to_insert = [
        {
            "timestamp": datetime.now().isoformat(),
            "model_version": f"v{version}",
            "epoch": epoch + 1,
            "accuracy": history['accuracy'][epoch],
            "val_accuracy": history['val_accuracy'][epoch],
            "loss": history['loss'][epoch],
            "val_loss": history['val_loss'][epoch]
        }
        for epoch in range(len(history['accuracy']))
    ]

    errors = bigquery_client.insert_rows_json(table_id, rows_to_insert)
    if errors == []:
        logger.debug("New rows have been added to BigQuery.")
    else:
        logger.error(f"Encountered errors while inserting rows: {errors}")

if __name__ == "__main__":
    logger.info("Starting model fine-tuning process")
    current_version = get_latest_model_version()
    new_version = increment_version(current_version)
    
    current_model_blob = f'm_net_v{current_version}.keras'
    data_prefix = 'correct_predicitons/'
    
    with tempfile.TemporaryDirectory() as model_temp_dir, tempfile.TemporaryDirectory() as data_temp_dir:
        current_model_path = os.path.join(model_temp_dir, 'current_model.keras')
        download_model_from_gcs(current_model_blob, current_model_path)
        
        download_data_from_gcs(data_prefix, data_temp_dir)
        
        new_model_path = os.path.join(model_temp_dir, 'new_model.keras')
        history = fine_tune_model(data_temp_dir, current_model_path, new_model_path)
        
        new_model_blob = f'm_net_v{new_version}.keras'
        upload_model_to_gcs(new_model_path, new_model_blob)
        logger.info(f"New model uploaded as version: v{new_version}")
        
        record_validation_data(new_version, history)
    
    logger.info("Model fine-tuning process completed")