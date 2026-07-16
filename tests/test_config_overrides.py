from llrops.config.loader import iter_program_calls, parse_set_overrides


def test_parse_set_overrides_native_types():
    overrides = parse_set_overrides([
        "flag=false",
        "count=5",
        "scale=0.25",
        "missing=null",
        "items=[1, 2, 3]",
        'mapping={"a": 1}',
        'text="001"',
        "raw=abc123",
    ])

    assert overrides == {
        "flag": False,
        "count": 5,
        "scale": 0.25,
        "missing": None,
        "items": [1, 2, 3],
        "mapping": {"a": 1},
        "text": "001",
        "raw": "abc123",
    }


def test_full_placeholder_substitution_preserves_override_type():
    config = {
        "variables": {"enabled": True, "n": 1},
        "programs": [
            {"program": "Dummy", "flag": "{enabled}", "count": "{n}"},
        ],
    }
    overrides = parse_set_overrides(["enabled=false", "n=7"])
    calls = list(iter_program_calls(config, overrides))
    _, program_config, _ = calls[0]

    assert program_config["flag"] is False
    assert program_config["count"] == 7
