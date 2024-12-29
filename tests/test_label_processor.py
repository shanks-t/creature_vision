# tests/test_label_processor.py
import pytest
from pathlib import Path
from src.preprocessing.label_processor import LabelProcessor


class TestLabelProcessor:
    @pytest.fixture
    def data_dir(self):
        return Path("./data")

    @pytest.fixture
    def label_processor(self):
        return LabelProcessor()

    def test_process_single_label_file(self, label_processor, data_dir):
        # Get first label file from correct_predictions
        label_path = next(data_dir.joinpath(
            "correct_predictions").glob("*_labels.json"))
        label_index = label_processor.process_label(label_path)

        # Basic validation
        assert isinstance(label_index, int)
        assert label_index >= 0

        # Test label mapping was created
        mapping = label_processor.get_label_mapping()
        assert len(mapping) > 0
        assert isinstance(mapping, dict)

    def test_label_consistency(self, label_processor, data_dir):
        # Process same label file twice to ensure consistent mapping
        label_files = list(data_dir.joinpath(
            "correct_predictions").glob("*_labels.json"))[:2]

        # Process first file twice
        first_index = label_processor.process_label(label_files[0])
        second_index = label_processor.process_label(label_files[0])

        assert first_index == second_index
