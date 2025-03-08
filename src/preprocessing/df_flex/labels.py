import json

from google.cloud import storage

LABEL_MAP_PATH = "processed/metadata/label_map.json"
LABEL_FILE_SUFFIX = "_labels.json"
LABEL_PREFIXES = ["correct_predictions/", "incorrect_predictions/"]


def load_label_map(bucket_name):
    """Load existing label map from GCS"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(LABEL_MAP_PATH)

    if blob.exists():
        return json.loads(blob.download_as_text())
    return {}  # Return empty if no label map exists


def save_label_map(bucket_name, label_map):
    """Save updated label map back to GCS"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(LABEL_MAP_PATH)
    blob.upload_from_string(json.dumps(label_map, indent=2))


def extract_label(file_path):
    """Extract label from file path (e.g., 'incorrect_predictions/whippet_12345_labels.json' → 'whippet')"""
    return file_path.split('/')[-1].split('_')[0]  # Extract first word in filename


def get_all_label_files(bucket_name):
    """Retrieve all label file paths from GCS bucket under specified prefixes"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    label_files = []
    for prefix in LABEL_PREFIXES:
        blobs = bucket.list_blobs(prefix=prefix)
        label_files.extend(
            [blob.name for blob in blobs if blob.name.endswith(LABEL_FILE_SUFFIX)])

    return label_files


def update_label_map(bucket_name):
    """Scan GCS filenames and update label map only if new labels are found"""
    label_files = get_all_label_files(bucket_name)

    # Load existing label map
    label_map = load_label_map(bucket_name)
    new_labels = set()

    for file_path in label_files:
        label = extract_label(file_path)
        if label not in label_map:
            new_labels.add(label)

    # If new labels found, update label map
    if new_labels:
        for label in new_labels:
            label_map[label] = len(label_map)

        save_label_map(bucket_name, label_map)
        print(f"Updated label map with {len(new_labels)} new labels.")
    else:
        print("No new labels found.")

# Example usage (this should be called from your pipeline)
# update_label_map("creature-vision-training-set")
