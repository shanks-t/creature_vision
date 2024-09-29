import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions
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
from prometheus_client import make_wsgi_app, Counter, Histogram
from werkzeug.middleware.dispatcher import DispatcherMiddleware

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

PREDICTIONS_TOTAL = Counter('predictions_total', 'Total number of predictions made')
CORRECT_PREDICTIONS = Counter('correct_predictions', 'Number of correct predictions')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Latency of predictions')

logger = structlog.get_logger()

app = Flask(__name__)


# Load the pre-trained MobileNetV2 model
model = load_model("./mobile_net_v3_small.keras")

# Initialize Google Cloud Storage client
storage_client = storage.Client(project="creature-vision")
bucket_name = 'creature-vision-training-set' 
bucket = storage_client.bucket(bucket_name)

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
    img_blob = bucket.blob(img_blob_name)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    img_blob.upload_from_string(img_byte_arr, content_type='image/jpeg')
    
    # Save labels to GCS
    label_blob_name = f"{directory}/{base_filename}_labels.json"
    label_blob = bucket.blob(label_blob_name)
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

@app.route('/predict', methods=['GET'])
def run_prediction():
    try:
        start_time = time.time()
        img_array, img, api_label = load_random_dog_image()
        
        with PREDICTION_LATENCY.time():
            predictions = model.predict(img_array)
        
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        model_label = decoded_predictions[0][1]
        
        is_correct = compare_breeds(api_label, model_label)
        if not is_correct:
            is_correct = fuzzy_match(api_label, model_label)
        
        PREDICTIONS_TOTAL.inc()
        if is_correct:
            CORRECT_PREDICTIONS.inc()
        
        result = {
            'status': 'correct' if is_correct else 'incorrect',
            'actual': api_label,
            'predicted': model_label
        }
        
        save_image_and_label(img, model_label, api_label, is_correct)
        
        logger.info("Prediction made",
                    actual=api_label,
                    predicted=model_label,
                    is_correct=is_correct,
                    latency=time.time() - start_time)
        
        return jsonify(result)
    
    except Exception as e:
        logger.exception("Error during prediction", error=str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return "Service is running", 200

# Add the metrics endpoint
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

if __name__ == "__main__":
    if tf.test.is_built_with_cuda():
        print("GPU acceleration enabled")
    else:
        print("Running on CPU or Metal")

    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))