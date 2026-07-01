from inference.main import compare_breeds, normalize_breed_name


def test_normalize_breed_name_applies_known_mapping():
    assert normalize_breed_name("alaskan-malamute") == {"malamute"}


def test_compare_breeds_uses_fuzzy_matching_for_base_model():
    assert compare_breeds("alaskan malamute", "malamute", "v3_0")


def test_compare_breeds_uses_exact_matching_for_custom_models():
    assert compare_breeds("malamute", "malamute", "v3_1")
    assert not compare_breeds("alaskan malamute", "malamute", "v3_1")
