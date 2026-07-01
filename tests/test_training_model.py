import pytest
from creature_vision_training.model import parse_model_version


def test_parse_model_version():
    assert parse_model_version("v3_12") == (3, 12)


def test_parse_model_version_rejects_bad_value():
    with pytest.raises(ValueError):
        parse_model_version("latest")
