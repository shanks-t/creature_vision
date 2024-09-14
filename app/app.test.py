import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import time
from google.cloud import storage
from fuzzywuzzy import fuzz
import os, json

if tf.test.is_built_with_cuda():
    # GpU acceleration available (Cloud Run)
    print("GPU acceleration enabled")
else:
    # Running on CPU or Metal (M2 MacBook)
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
    # Determine the directory based on correctness
    directory = "../data/correct_predictions" if is_correct else "../data/incorrect_predictions"
    
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Generate a unique filename using timestamp
    timestamp = int(time.time())
    if is_correct:
        base_filename = f"{model_label}_{timestamp}"
    else:
        base_filename = f"{api_label}_{timestamp}"   
    # Save the image
    img_filename = f"{base_filename}.jpg"
    img_path = os.path.join(directory, img_filename)
    img.save(img_path, format='JPEG')
    
   # Save both labels to a JSON file
    label_filename = f"{base_filename}_labels.json"
    label_path = os.path.join(directory, label_filename)
    label_data = {
        "model_label": model_label,
        "api_label": api_label
    }
    with open(label_path, 'w') as f:
        json.dump(label_data, f)
    
    print(f"Saved image to {img_path}")
    print(f"Saved label to {label_path}")

# def save_image_and_label(img, breed, is_correct):
#     directory = "correct_predictions" if is_correct else "incorrect_predictions"
#     blob_name = f"{directory}/{breed}_{int(time.time())}.jpg"
#     blob = bucket.blob(blob_name)
#     img_byte_arr = BytesIO()
#     img.save(img_byte_arr, format='JPEG')
#     img_byte_arr = img_byte_arr.getvalue()
#     blob.upload_from_string(img_byte_arr, content_type='image/jpeg')
    
#     label_blob = bucket.blob(f"{blob_name}.txt")
#     label_blob.upload_from_string(breed)

def normalize_breed_name(breed):
    return set(breed.lower().replace('-', ' ').replace('_', ' ').split())

EXCLUDE_WORDS = {'dog', 'hound', 'terrier', 'shepherd', 'retriever', 'spaniel', 'poodle'}

def compare_breeds(actual, predicted, strict_match_threshold=0.75):
    print(f"actual: {actual}")
    print(f"predicted: {predicted}")
    actual_set = normalize_breed_name(actual)
    predicted_set = normalize_breed_name(predicted)
    print(f"Actual set: {actual_set}")
    print(f"Predicted set: {predicted_set}")

    # Check for exact match
    if actual_set == predicted_set:
        print("Exact match found")
        return True

    # Check for strict partial match
    common_words = actual_set.intersection(predicted_set)
    
    # Handle empty sets after exclusion
    if not actual_set or not predicted_set:
        return actual.lower() == predicted.lower()

    actual_coverage = len(common_words) / len(actual_set)
    predicted_coverage = len(common_words) / len(predicted_set)
    
    if (actual_coverage >= strict_match_threshold and predicted_coverage >= strict_match_threshold):
        print(f"Strict partial match found: {common_words}")
        return True

    # Handle cases where one set is a subset of the other
    if actual_set.issubset(predicted_set) or predicted_set.issubset(actual_set):
        print(f"Subset match found: {common_words}")
        return True

    return False

def fuzzy_match(actual, predicted, threshold=85):
    print("first string comparison false. Running fuzzy match...")
    ratio = fuzz.ratio(actual.lower(), predicted.lower())
    print(f"Fuzzy match ratio: {ratio}")
    return ratio >= threshold

def run_prediction(event, context):
    try:
        img_array, img, api_label = load_random_dog_image()
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        
       # Get top predicted breed (highest confidence)
        model_label = decoded_predictions[0][1]
        
        # First, try exact matching (ignoring order)
        is_correct = compare_breeds(api_label, model_label)
        print(f"Exact match result: {is_correct}")
        
        # If not correct, try fuzzy matching
        if not is_correct:
            is_correct = fuzzy_match(api_label, model_label)
            print(f"Fuzzy match result: {is_correct}")
        
        if is_correct:
            print(f"Correct prediction: {api_label}")
            result = {'status': 'correct', 'actual': api_label, 'predicted': model_label}
            # if the prediciton is corract preserve the models label prediction and add to dataset
            save_image_and_label(img, model_label, api_label, is_correct)
        else:
            print(f"Incorrect prediction. Actual: {api_label}, Predicted: {model_label}")
            result = {'status': 'incorrect', 'actual': api_label, 'predicted': model_label}
            # if the prediction is incorrect add the data with the correct label to dataset
            save_image_and_label(img, model_label, api_label, is_correct)
        
        # Here you could log the result to Cloud Logging or save to a database
        print(f"Prediction result: {result}")
        
        return result
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {'error': str(e)}

# This is for local testing
if __name__ == "__main__":
    print(run_prediction(None, None))