import jijmodeling as jm
import pytest


@pytest.fixture
def jm_sampleset() -> jm.SampleSet:
    return jm.SampleSet(
        **{
            "record": jm.Record(
                **{
                    "solution": {
                        "x": [
                            (([0, 1], [0, 1]), [1, 1], (2, 2)),
                            (([0, 1], [1, 0]), [1, 1], (2, 2)),
                            (([], []), [], (2, 2)),
                            (([0, 1], [0, 0]), [1, 1], (2, 2)),
                        ]
                    },
                    "num_occurrences": [4, 3, 2, 1],
                }
            ),
            "evaluation": jm.Evaluation(
                **{
                    "objective": [3.0, 24.0, 0.0, 17.0],
                    "constraint_violations": {
                        "onehot1": [0.0, 0.0, 2.0, 0.0],
                        "onehot2": [0.0, 0.0, 2.0, 2.0],
                    },
                    "constraint_values": [
                        {"onehot1": [0.0, 0.0], "onehot2": [0.0, 0.0]},
                        {"onehot1": [0.0, 0.0], "onehot2": [0.0, 0.0]},
                        {"onehot1": [1.0, 1.0], "onehot2": [1.0, 1.0]},
                        {"onehot1": [0.0, 0.0], "onehot2": [1.0, 1.0]},
                    ],
                    "constraint_forall": {"onehot1": [[0], [1]], "onehot2": [[0], [1]]},
                    "penalty": {},
                }
            ),
            "measuring_time": jm.MeasuringTime(
                **{"solve": None, "system": None, "total": None}
            ),
        }
    )


@pytest.fixture
def jm_sampleset_no_constraint() -> jm.SampleSet:
    return jm.SampleSet(
        **{
            "record": jm.Record(
                **{
                    "solution": {
                        "x": [
                            (([0, 1], [0, 1]), [1, 1], (2, 2)),
                            (([0, 1], [1, 0]), [1, 1], (2, 2)),
                            (([], []), [], (2, 2)),
                            (([0, 1], [0, 0]), [1, 1], (2, 2)),
                        ]
                    },
                    "num_occurrences": [4, 3, 2, 1],
                }
            ),
            "evaluation": jm.Evaluation(
                **{
                    "objective": [3.0, 24.0, 0.0, 17.0],
                    "constraint_violations": {},
                    "penalty": {},
                }
            ),
            "measuring_time": jm.MeasuringTime(
                **{"solve": None, "system": None, "total": None}
            ),
        }
    )


@pytest.fixture
def jm_sampleset_no_obj_no_constraint() -> jm.SampleSet:
    return jm.SampleSet(
        **{
            "record": jm.Record(
                **{
                    "solution": {
                        "x": [
                            (([0, 1], [0, 1]), [1, 1], (2, 2)),
                            (([0, 1], [1, 0]), [1, 1], (2, 2)),
                            (([], []), [], (2, 2)),
                            (([0, 1], [0, 0]), [1, 1], (2, 2)),
                        ]
                    },
                    "num_occurrences": [4, 3, 2, 1],
                }
            ),
            "evaluation": jm.Evaluation(
                **{
                    "constraint_violations": {},
                    "penalty": {},
                }
            ),
            "measuring_time": jm.MeasuringTime(
                **{"solve": None, "system": None, "total": None}
            ),
        }
    )
