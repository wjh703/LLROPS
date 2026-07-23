import llrops.cli as cli

from llrops.programs.registry import available_programs, program, run_program


def test_program_registry_is_case_insensitive():
    @program("TestCanonicalProgram")
    def canonical(config, context):
        return config["value"]

    assert "TestCanonicalProgram" in available_programs()
    assert run_program("testcanonicalprogram", {"value": 3}, None) == 3


def test_program_discovery_registers_every_configurable_program():
    cli._import_programs()

    assert {
        "CrdToMini",
        "LlrAdjustment",
        "LlrNormalEquations",
        "LlrResiduals",
        "NormalPointsToLlrops",
        "NormalsCombineSolve",
    } <= set(available_programs())
