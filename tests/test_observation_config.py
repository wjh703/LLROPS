import pytest

from lunarops.classes.observation_factory import validate_observation_config


def test_observation_config_needs_no_uncertainty_selector():
    validate_observation_config({"inputFilesNormalPoints": ["input.txt.gz"]})


@pytest.mark.parametrize("key", ["uncertainty", "uncertaintyModel"])
def test_removed_program_uncertainty_selectors_are_rejected(key):
    with pytest.raises(ValueError, match="uncertainty_two_way_s"):
        validate_observation_config({key: "obsolete"})


def test_removed_global_uncertainty_selector_is_rejected():
    with pytest.raises(ValueError, match="globals"):
        validate_observation_config({}, {"uncertaintyModel": "obsolete"})
