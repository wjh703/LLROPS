from uuid import uuid4

import pytest

from lunarops.config.registry import (
    DuplicateClassRegistrationError,
    create,
    register_factory,
)


def test_factory_replacement_must_be_explicit():
    category = f"test_registry_{uuid4().hex}"

    def original(config, context):
        return "original"

    def replacement(config, context):
        return "replacement"

    register_factory(category, "model", original)
    with pytest.raises(DuplicateClassRegistrationError, match="replace=True"):
        register_factory(category, "model", replacement)

    register_factory(category, "model", replacement, replace=True)
    assert create(category, "model") == "replacement"
