import json


class LabelProcessor:
    def __init__(self):
        self.label_mapping = {}  # To store unique labels and their indices
        self.reverse_mapping = {}  # To store indices to labels mapping

    def process_label(self, label_input):
        if isinstance(label_input, dict):
            actual_label = label_input['api_label']
        else:
            with open(label_input, 'r') as f:
                label_data = json.load(f)
                actual_label = label_data['api_label']

        # Map the label to a numeric index if it's not already in our mapping
        if actual_label not in self.label_mapping:
            self.label_mapping[actual_label] = len(self.label_mapping)

        return self.label_mapping[actual_label]

    def get_class_names(self) -> list:
        """
        Returns ordered list of class names based on their numeric indices.

        Returns:
            List of class names ordered by their corresponding numeric indices
        """
        # Ensure all indices exist in reverse_mapping
        if not all(i in self.reverse_mapping for i in range(len(self.label_mapping))):
            # Rebuild reverse mapping if necessary
            self.reverse_mapping = {v: k for k,
                                    v in self.label_mapping.items()}

        return [self.reverse_mapping[idx] for idx in range(len(self.label_mapping))]

    def get_num_classes(self):
        return len(self.label_mapping)

    def get_label_mapping(self):
        return self.label_mapping
