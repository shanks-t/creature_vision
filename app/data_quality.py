import os
import json
from fuzzywuzzy import fuzz
import argparse

def load_training_labels(process_all=False):
    training_labels_path = "../data/meta/ImageNetLabels.txt"
    with open(training_labels_path, 'r') as f:
        all_labels = set(line.strip().lower() for line in f)

    if process_all:
        return all_labels

    # If not processing all, exclude labels that have already been checked
    incorrect_predictions_dir = "../data/incorrect_predictions"
    processed_labels = set()

    for filename in os.listdir(incorrect_predictions_dir):
        if filename.endswith("_labels.json"):
            file_path = os.path.join(incorrect_predictions_dir, filename)
            with open(file_path, 'r') as f:
                label_data = json.load(f)
            if 'is_in_training_set' in label_data:
                processed_labels.add(label_data['api_label'].lower())

    return all_labels - processed_labels

EXCLUDE_WORDS = {'dog'}

def normalize_breed_name(breed):
    words = breed.lower().replace('-', ' ').replace('_', ' ').split()
    return set(word for word in words if word not in EXCLUDE_WORDS)

def compare_breeds(actual, predicted, strict_match_threshold=0.75):
    actual_set = normalize_breed_name(actual)
    predicted_set = normalize_breed_name(predicted)

    if actual_set == predicted_set:
        return True

    common_words = actual_set.intersection(predicted_set)
    
    if not actual_set or not predicted_set:
        return False

    actual_coverage = len(common_words) / len(actual_set)
    predicted_coverage = len(common_words) / len(predicted_set)
    
    if (actual_coverage >= strict_match_threshold and predicted_coverage >= strict_match_threshold):
        return True

    if actual_set.issubset(predicted_set) or predicted_set.issubset(actual_set):
        return True

    return False

def fuzzy_match(actual, predicted, threshold=85):
    actual_filtered = ' '.join(word for word in actual.lower().split() if word not in EXCLUDE_WORDS)
    predicted_filtered = ' '.join(word for word in predicted.lower().split() if word not in EXCLUDE_WORDS)
    
    ratio = fuzz.token_sort_ratio(actual_filtered, predicted_filtered)
    return ratio >= threshold

def check_incorrect_predictions(process_all=False):
    incorrect_predictions_dir = "../data/incorrect_predictions"
    training_labels = load_training_labels(process_all)

    for filename in os.listdir(incorrect_predictions_dir):
        if filename.endswith("_labels.json"):
            file_path = os.path.join(incorrect_predictions_dir, filename)
            with open(file_path, 'r') as f:
                label_data = json.load(f)

            if not process_all and 'is_in_training_set' in label_data:
                continue

            api_label = label_data['api_label']
            is_in_training_set = False

            for training_label in training_labels:
                if compare_breeds(api_label, training_label) or fuzzy_match(api_label, training_label):
                    is_in_training_set = True
                    break

            label_data['is_in_training_set'] = is_in_training_set
            with open(file_path, 'w') as f:
                json.dump(label_data, f, indent=2)

            print(f"Processed {filename}: API Label: {api_label}, In Training Set: {is_in_training_set}")

    print("Finished processing incorrect predictions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process incorrect predictions.")
    parser.add_argument('--all', action='store_true', help='Process all files, including those already cleaned')
    args = parser.parse_args()

    check_incorrect_predictions(process_all=args.all)