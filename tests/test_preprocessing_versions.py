from df_flex.pipeline import get_previous_version, parse_model_version


def test_parse_model_version():
    assert parse_model_version("v3_12") == (3, 12)
    assert parse_model_version("local-dev") is None


def test_get_previous_version():
    assert get_previous_version("v3_12") == "v3_11"
    assert get_previous_version("v3_0") is None
    assert get_previous_version("local-dev") is None
