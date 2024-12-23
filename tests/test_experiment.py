import jijmodeling as jm
import ommx_pyscipopt_adapter as scip_ad

import minto


def knapsack():
    v = jm.Placeholder("v", ndim=1)
    n = v.shape[0]
    w = jm.Placeholder("w", shape=(n,))
    C = jm.Placeholder("C")
    x = jm.BinaryVar("x", shape=(n,))

    problem = jm.Problem("knapsack", sense=jm.ProblemSense.MAXIMIZE)
    i = jm.Element("i", n)
    problem += jm.sum(i, v[i] * x[i])
    problem += jm.Constraint("capa", jm.sum(i, w[i] * x[i]) <= C)
    return problem


def knapsack_data():
    return {"v": [1, 2, 3, 4, 5], "w": [2, 3, 4, 5, 6], "C": 8}


def knapsack_instance():
    problem = knapsack()
    data = knapsack_data()
    interpreter = jm.Interpreter(data)
    instance = interpreter.eval_problem(problem)
    return instance


def knapsack_solution():
    instance = knapsack_instance()
    scip_model = scip_ad.instance_to_model(instance)
    scip_model.optimize()
    solution = scip_ad.model_to_solution(scip_model, instance)
    return solution


def test_experiment(tmp_path):
    exp = minto.Experiment(savedir=tmp_path)
    problem = knapsack()
    data = knapsack_data()
    instance = knapsack_instance()
    solution = knapsack_solution()

    exp.log_problem("knapsack", problem)
    exp.log_object("knapsack", data)
    exp.log_instance("knapsack", instance)

    num_iter = 3
    for i in range(num_iter):
        with exp.run():
            exp.log_parameter("value", i)
            exp.log_solution("knapsack", solution)

    exp2 = minto.Experiment.load_from_dir(tmp_path / exp.experiment_name)
    assert exp2.experiment_name == exp.experiment_name
    assert jm.is_same(exp2.dataspace.experiment_datastore.problems["knapsack"], problem)
    assert exp2.dataspace.experiment_datastore.objects["knapsack"] == data
    assert exp2.dataspace.experiment_datastore.instances["knapsack"] == instance
    for i in range(num_iter):
        assert exp2.dataspace.run_datastores[i].parameters["value"] == i
        assert exp2.dataspace.run_datastores[i].solutions["knapsack"] == solution
