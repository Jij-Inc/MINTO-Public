import ommx.v1 as ommx

import minto


def test_v1_0_load_data():
    """
    Test Data is created by the following code:
    ```python
    instance_name = "reblock115"
    instance = miplib2017(instance_name)
    timelimit_list = [0.1, 0.5, 1, 2]

    experiment = minto.Experiment('scip_exp', auto_saving=False, savedir='./tests/exp_data')
    experiment.log_instance(instance_name, instance)
    scip_model = scip_ad.instance_to_model(instance)

    for timelimit in timelimit_list:
        with experiment.run():
            experiment.log_parameter('timelimit', timelimit)

            scip_model.setParam("limits/time", timelimit)
            scip_model.optimize()
            solution = scip_ad.model_to_solution(scip_model, instance)

            experiment.log_solution("scip", solution)
    ```
    """
    timelimit_list = [0.1, 0.5, 1, 2]

    load_data = "tests/exp_data/scip_exp_20250112073656"
    experiment = minto.Experiment.load_from_dir(load_data)
    exp_instances = experiment.dataspace.experiment_datastore.instances
    assert "reblock115" in exp_instances

    run_data = experiment.dataspace.run_datastores
    assert len(run_data) == len(timelimit_list)
    for i in range(len(run_data)):
        assert run_data[i].parameters == {"timelimit": timelimit_list[i]}

        solution = run_data[i].solutions["scip"]
        assert isinstance(solution, ommx.Solution)
