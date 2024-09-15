import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import time
from google.cloud import storage
from fuzzywuzzy import fuzz
import os
import json
from flask import Flask, jsonify

app = Flask(__name__)

if tf.test.is_built_with_cuda():
    print("GPU acceleration enabled")
else:
    print("Running on CPU or Metal")

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Initialize Google Cloud Storage client
storage_client = storage.Client()
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
        img_array, img, api_label = load_random_dog_image()
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        
        # Get top predicted breed (highest confidence)
        model_label = decoded_predictions[0][1]
        
        # First, try exact matching (ignoring order)
        is_correct = compare_breeds(api_label, model_label)
        
        # If not correct, try fuzzy matching
        if not is_correct:
            is_correct = fuzzy_match(api_label, model_label)
        
        result = {
            'status': 'correct' if is_correct else 'incorrect',
            'actual': api_label,
            'predicted': model_label
        }
        
        # Save image and label
        save_image_and_label(img, model_label, api_label, is_correct)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return "Service is running", 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))