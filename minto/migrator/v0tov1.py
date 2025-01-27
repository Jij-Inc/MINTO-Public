import jijmodeling as jm

import minto.v0


def migrate_to_v1_from_v0(experiment: minto.v0.Experiment) -> minto.Experiment:
    df = experiment.table(enable_sampleset_expansion=False)
    exp_v1 = minto.Experiment(name=experiment.name)
    ignored_key = ["experiment_name", "run_id"]
    for row_id, row in df.iterrows():
        with exp_v1.run():
            for name, value in row.items():
                if name in ignored_key:
                    continue
                if isinstance(value, (str, int, float)):
                    exp_v1.log_parameter(name, value)
                elif isinstance(value, jm.Problem):
                    exp_v1.log_problem(name, value)
                elif isinstance(value, jm.experimental.SampleSet):
                    exp_v1.log_sampleset(name, value)
                elif isinstance(value, list):
                    exp_v1.log_object(name, {name: value})
                elif isinstance(value, dict):
                    exp_v1.log_object(name, value)
    return exp_v1
