import logging

from llrops.lifecycle import close_resources
from llrops.parallel.cache import close_cached_objects
from llrops.programs.base import available_programs, program, program_alias, run_program


class Resource:
    def __init__(self):
        self.count = 0

    def close(self):
        self.count += 1


def test_close_resources_deduplicates_resources(caplog):
    resource = Resource()
    with caplog.at_level(logging.WARNING):
        close_resources([resource, resource], owner="test")
    assert resource.count == 1
    resource.count = 0
    close_cached_objects({"direct": resource, "nested": {"same": resource}})
    assert resource.count == 1


def test_program_alias_is_not_a_second_public_program():
    @program("TestCanonicalProgram")
    def canonical(config, context):
        return config["value"]

    program_alias("TestLegacyProgram", "TestCanonicalProgram")
    assert "TestCanonicalProgram" in available_programs()
    assert "TestLegacyProgram" not in available_programs()
    assert run_program("TestLegacyProgram", {"value": 3}, None) == 3
