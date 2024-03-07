import jijmodeling as jm
import numpy as np

import minto


def test_simple_case():
    experiment = minto.Experiment()
    data: list[dict[str, np.ndarray]] = [
        {"x": np.array([0.0, 1.0])},
        {"x": np.array([1.0, 0.0])},
    ]
    sampleset = jm.experimental.SampleSet.from_array(data)
    with experiment.run():
        experiment.log_result("result", sampleset)
    table = experiment.table()
    assert len(table) == len(sampleset)
