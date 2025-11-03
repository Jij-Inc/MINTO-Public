"""
Helper functions for Instance, Solution, and SampleSet comparison
"""

import ommx.v1


def assert_instance_equal(
    instance: ommx.v1.Instance, expected_instance: ommx.v1.Instance
):
    """
    Helper function for comparison between Instance objects

    Args:
        instance: Instance to compare
        expected_instance: Expected Instance
    """
    assert instance.to_bytes() == expected_instance.to_bytes()


def assert_solution_equal(
    solution: ommx.v1.Solution, expected_solution: ommx.v1.Solution
):
    """
    Helper function for comparison between Solution objects

    Args:
        solution: Solution to compare
        expected_solution: Expected Solution
    """
    assert type(solution) is ommx.v1.Solution
    assert solution.state.entries == expected_solution.state.entries
    assert solution.objective == expected_solution.objective
    assert len(solution.decision_variables) == len(expected_solution.decision_variables)
    for i, (dv1, dv2) in enumerate(
        zip(solution.decision_variables, expected_solution.decision_variables)
    ):
        assert dv1.id == dv2.id
        assert dv1.value == dv2.value
    assert solution.feasible == expected_solution.feasible
    assert solution.optimality == expected_solution.optimality


def assert_sampleset_equal(
    sampleset: ommx.v1.SampleSet, expected_sampleset: ommx.v1.SampleSet
):
    """
    Helper function for comparison between SampleSet objects

    Args:
        sampleset: SampleSet to compare
        expected_sampleset: Expected SampleSet
    """
    assert type(sampleset) is ommx.v1.SampleSet
    assert sampleset.objectives == expected_sampleset.objectives
    assert len(sampleset.decision_variables) == len(
        expected_sampleset.decision_variables
    )
    for i, (dv1, dv2) in enumerate(
        zip(sampleset.decision_variables, expected_sampleset.decision_variables)
    ):
        assert dv1.id == dv2.id
        assert dv1.to_bytes() == dv2.to_bytes()
    assert sampleset.feasible == expected_sampleset.feasible
