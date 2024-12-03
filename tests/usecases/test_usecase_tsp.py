import jijmodeling as jm
import jijmodeling_transpiler as jmt
import numpy as np
import openjij as oj

import minto


def random_tsp(n: int):
    x, y = np.random.uniform(0, 1, (2, n))
    XX, XX_T = np.meshgrid(x, x)
    YY, YY_T = np.meshgrid(y, y)

    distance = np.sqrt((XX - XX_T) ** 2 + (YY - YY_T) ** 2)

    return (x, y), distance


def test_tsp_param_search():
    # TSP
    d = jm.Placeholder("d", ndim=2)
    n = d.shape[0]
    n.set_latex("n")

    x = jm.BinaryVar("x", shape=(n, n))

    problem = jm.Problem("TSP")
    i, j, t = (
        jm.Element("i", belong_to=n),
        jm.Element("j", belong_to=n),
        jm.Element("t", belong_to=n),
    )
    problem += jm.sum([i, j, t], d[i, j] * x[i, t] * x[j, (t + 1) % n])
    problem += jm.Constraint("one-city", jm.sum(i, x[i, t]) == 1, forall=t)
    problem += jm.Constraint("one-time", jm.sum(t, x[i, t]) == 1, forall=i)

    _, distance = random_tsp(n=5)

    compiled_instance = jmt.core.compile_model(problem, instance={"d": distance})

    pubo_builder = jmt.core.pubo.transpile_to_pubo(compiled_instance)
    qubo, _ = pubo_builder.get_qubo_dict()

    sampler = oj.SASampler()
    response = sampler.sample_qubo(qubo, num_reads=5)

    sampleset = jmt.core.pubo.decode_from_openjij(
        response, pubo_builder, compiled_instance
    )
    sampleset = jm.experimental.from_old_sampleset(sampleset)

    experiment = minto.Experiment()
    with experiment.run():
        experiment.log_result("result", sampleset)
    experiment.table()
