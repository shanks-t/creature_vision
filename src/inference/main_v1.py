import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import time
from google.cloud import storage
from fuzzywuzzy import fuzz
import Levenshtein
import os
import json
from flask import Flask, jsonify
import structlog
import tempfile
from google.cloud import bigquery
from datetime import datetime

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

app = Flask(__name__)

bigquery_client = bigquery.Client(project="creature-vision")
storage_client = storage.Client(project="creature-vision")

data_bucket_name = 'creature-vision-training-set' 
data_bucket = storage_client.bucket(data_bucket_name)


model_version = os.getenv('MODEL_VERSION')
print(f"model version: {model_version}")
if model_version is None:
    raise ValueError("MODEL_VERSION environment variable is not set")
# model_name = "m_net_" + model_version

# Load the pre-trained MobileNetV3 model
model = load_model("./model.keras")

def insert_prediction_data(actual, predicted, is_correct, latency, confidence):
    dataset_id = "dog_prediction_app"
    table_id = "prediction_metrics"
    table_ref = bigquery_client.dataset(dataset_id).table(table_id)
    table = bigquery_client.get_table(table_ref)

    rows_to_insert = [{
        "timestamp": datetime.now(),
        "actual": actual,
        "predicted": predicted,
        "is_correct": is_correct,
        "latency": latency,
        "confidence": confidence,
        "model_version": model_version
    }]

    errors = bigquery_client.insert_rows(table, rows_to_insert)
    if errors:
        logger.error(f"Encountered errors while inserting rows: {errors}")

def load_random_dog_image():
    dog_api_url = "https://dog.ceo/api/breeds/image/random"
    
    try:
        response = requests.get(dog_api_url)
        response.raise_for_status()
        data = response.json()
        if data['status'] == 'success':
            image_url = data['message']
            breed = image_url.split('/')[4].replace('-', ' ')
        else:
            raise ValueError("Failed to fetch dog image from the API")
        
        img_response = requests.get(image_url)
        img_response.raise_for_status()
        img = Image.open(BytesIO(img_response.content))
        img = img.convert('RGB')
        img = img.resize((224, 224))
        
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        print(f"breed from api: {breed}")
        
        return img_array, img, breed
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch dog image: {str(e)}")

def save_image_and_label(img, model_label, api_label, is_correct):
    directory = "correct_predictions" if is_correct else "incorrect_predictions"
    timestamp = int(time.time())
    base_filename = f"{model_label}_{timestamp}" if is_correct else f"{api_label}_{timestamp}"
    
    # Save image to GCS
    img_blob_name = f"{directory}/{base_filename}.jpg"
    img_blob = data_bucket.blob(img_blob_name)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    img_blob.upload_from_string(img_byte_arr, content_type='image/jpeg')
    
    # Save labels to GCS
    label_blob_name = f"{directory}/{base_filename}_labels.json"
    label_blob = data_bucket.blob(label_blob_name)
    label_data = {
        "model_label": model_label,
        "api_label": api_label
    }
    label_blob.upload_from_string(json.dumps(label_data), content_type='application/json')

def normalize_breed_name(breed):
    return set(breed.lower().replace('-', ' ').replace('_', ' ').split())

EXCLUDE_WORDS = {'dog', 'hound', 'terrier', 'shepherd', 'retriever', 'spaniel', 'poodle'}

def compare_breeds(actual, predicted, strict_match_threshold=0.75):
    actual_set = normalize_breed_name(actual)
    predicted_set = normalize_breed_name(predicted)
    
    # Check for exact match
    if actual_set == predicted_set:
        return True

    # Check for strict partial match
    common_words = actual_set.intersection(predicted_set)
    
    # Handle empty sets after exclusion
    if not actual_set or not predicted_set:
        return actual.lower() == predicted.lower()

    actual_coverage = len(common_words) / len(actual_set)
    predicted_coverage = len(common_words) / len(predicted_set)
    
    if (actual_coverage >= strict_match_threshold and predicted_coverage >= strict_match_threshold):
        return True

    # Handle cases where one set is a subset of the other
    if actual_set.issubset(predicted_set) or predicted_set.issubset(actual_set):
        return True

    return False

def fuzzy_match(actual, predicted, threshold=85):
    ratio = fuzz.ratio(actual.lower(), predicted.lower())
    return ratio >= threshold

@app.route('/predict/', methods=['GET'])
@app.route('/predict', methods=['GET'])
def run_prediction():
    try:
        start_time = time.time()
        img_array, img, api_label = load_random_dog_image()
        
        predictions = model.predict(img_array)
        
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        model_label = decoded_predictions[0][1]
        confidence = float(decoded_predictions[0][2])
        
        is_correct = compare_breeds(api_label, model_label)
        if not is_correct:
            is_correct = fuzzy_match(api_label, model_label)
        
        latency = time.time() - start_time 
        
        result = {
            'model_version': model_version,
            'is_correct': is_correct,
            'actual': api_label,
            'predicted': model_label,
            'confidence': confidence,
            'latency': latency
        }
        
        save_image_and_label(img, model_label, api_label, is_correct)
        # Insert prediction data into BigQuery
        insert_prediction_data(api_label, model_label, is_correct, latency, confidence)
        
        logger.info("Prediction made",
                    model_version=model_version,
                    actual=api_label,
                    predicted=model_label,
                    is_correct=is_correct,
                    confidence=confidence,
                    latency=latency)
        
        return jsonify(result)
    
    except Exception as e:
        logger.exception("Error during prediction", error=str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return "Service is running", 200

if __name__ == "__main__":
    if tf.test.is_built_with_cuda():
        print("GPU acceleration enabled")
    else:
        print("Running on CPU or Metal")

    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))