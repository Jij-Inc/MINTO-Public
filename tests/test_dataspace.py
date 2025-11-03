import jijmodeling as jm
import ommx.artifact as ox_art
import ommx_pyscipopt_adapter as scip_ad
from helpers import assert_instance_equal, assert_solution_equal

from minto.v1.datastore import DataStore
from minto.v1.exp_dataspace import ExperimentDataSpace


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
    adapter = scip_ad.OMMXPySCIPOptAdapter(instance)
    scip_model = adapter.solver_input
    scip_model.optimize()
    solution = adapter.decode(scip_model)
    return solution


def test_simple_case(tmp_path):
    ds = ExperimentDataSpace("test")
    problem = knapsack()
    data = knapsack_data()
    instance = knapsack_instance()
    solution = knapsack_solution()

    ds.add_exp_data("knapsack", problem, "problems", with_save=True, save_dir=tmp_path)

    run_id = ds.add_run_datastore(DataStore())
    ds.add_run_data(
        run_id, "knapsack", data, "objects", with_save=True, save_dir=tmp_path
    )
    ds.add_run_data(
        run_id, "knapsack", instance, "instances", with_save=True, save_dir=tmp_path
    )
    ds.add_run_data(
        run_id, "value", 0.2, "parameters", with_save=True, save_dir=tmp_path
    )
    ds.add_run_data(
        run_id, "sol", solution, "solutions", with_save=True, save_dir=tmp_path
    )
    ds.add_run_data(
        run_id, "meta", "meta", "meta_data", with_save=True, save_dir=tmp_path
    )
    ds2 = ExperimentDataSpace.load_from_dir(tmp_path)
    assert jm.is_same(ds2.experiment_datastore.problems["knapsack"], problem)
    assert len(ds2.run_datastores) == 1

    run_id = 0
    datastore = ds2.run_datastores[run_id]
    assert datastore.objects["knapsack"] == data
    assert_instance_equal(datastore.instances["knapsack"], instance)
    assert datastore.parameters == {"value": 0.2}
    assert datastore.meta_data == {"meta": "meta", "run_id": run_id}
    assert_solution_equal(datastore.solutions["sol"], solution)

    artifact_path = tmp_path / "sample.ommx"
    builder = ox_art.ArtifactBuilder.new_archive_unnamed(artifact_path)
    ds2.add_to_artifact_builder(builder)
    builder.build()

    ds3 = ExperimentDataSpace.load_from_ommx_archive(artifact_path)
    assert jm.is_same(ds3.experiment_datastore.problems["knapsack"], problem)
    assert len(ds3.run_datastores) == 1
    run_id = 0
    datastore = ds3.run_datastores[run_id]
    assert datastore.objects["knapsack"] == data
    instance.annotations.update(
        {
            "org.minto.run_id": "0",
            "org.minto.space": "run",
            "org.minto.name": "knapsack",
            "org.minto.storage": "instances",
        }
    )
    assert_instance_equal(datastore.instances["knapsack"], instance)
    assert datastore.parameters == {"value": 0.2}
    assert datastore.meta_data == {"meta": "meta", "run_id": run_id}
    solution.annotations.update(
        {
            "org.minto.run_id": "0",
            "org.minto.space": "run",
            "org.minto.name": "sol",
            "org.minto.storage": "solutions",
        }
    )
    assert_solution_equal(datastore.solutions["sol"], solution)
