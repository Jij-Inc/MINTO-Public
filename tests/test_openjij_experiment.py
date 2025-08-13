# %%
import random

import jijmodeling as jm
import networkx as nx
import ommx_openjij_adapter as oj_ad
import openjij as oj

import minto


def max_cut():
    n = jm.Placeholder("n")
    E = jm.Placeholder("E", ndim=2)
    x = jm.BinaryVar("x", shape=(n,))
    problem = jm.Problem("max-cut", sense=jm.ProblemSense.MAXIMIZE)

    e = jm.Element("e", E)
    i, j = e[0], e[1]
    i.set_latex("i")
    j.set_latex("j")
    problem += jm.sum(e, x[i] + x[j] - 2 * x[i] * x[j])

    return problem


def test_openjij_maxcut():
    G = nx.Graph()
    n = 30
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                G.add_edge(i, j)

    instance_data = {"E": [edge for edge in G.edges], "n": n}
    instance = jm.Interpreter(instance_data).eval_problem(max_cut())

    # %%
    qubo, _ = instance.to_qubo()
    oj_adapter = oj_ad.OMMXOpenJijSAAdapter(instance)

    # %%
    experiment = minto.Experiment(auto_saving=False)

    num_sweeps_list = [500, 1000, 3000, 5000, 10000]
    beta_set = [(None, None), (0.1, 100)]
    sampler = oj.SASampler()

    for beta_min, beta_max in beta_set:
        for num_sweeps in num_sweeps_list:
            run = experiment.run()
            with run:
                print(
                    f"num_sweeps: {num_sweeps}, beta_min: {beta_min}, "
                    f"beta_max: {beta_max}"
                )
                run.log_parameter("num_sweeps", num_sweeps)
                response = sampler.sample_qubo(
                    qubo,
                    num_reads=100,
                    num_sweeps=num_sweeps,
                    beta_min=beta_min,
                    beta_max=beta_max,
                )
                sampleset = oj_adapter.decode_to_sampleset(response)
                run.log_sampleset(sampleset)
                run.log_params({"beta_max": beta_max, "beta_min": beta_min})
                run.log_parameter("time", response.info["sampling_time"] * (1e-6))

    # %%
    df = experiment.get_run_table()

    # %%
    df

    # %%
    # extract the nan records
    # df_default = df[df[("parameter", "beta_min")].isna()]
    # df_tuned = df[df[("parameter", "beta_min")].notna()]

    # %%
    # This section is commented out because it requires matplotlib
    # and should only be run interactively in a notebook environment

    # import matplotlib.pyplot as plt
    #
    # time_key = ("parameter", "time")
    #
    # plt.plot(
    #     df_default[("sampleset_0", "obj_mean")],
    #     df_default[time_key],
    #     label="default",
    # )
    # plt.plot(
    #     df_tuned[("sampleset_0", "obj_mean")],
    #     df_tuned[time_key],
    #     label=r"tuned ($\beta_{\min}=0.01, \beta_{\max}=100$)",
    # )
    # plt.yscale("log")
    # plt.xlabel("obj_mean")
    # plt.ylabel("elapsed_time [s]; num_reads=100")
    # plt.title(r"Performance Comparison of Default vs Tuned $\beta$")
    # plt.legend()
