import jijmodeling as jm
import numpy as np

from .problem import Problem


class KnapsackProblem(Problem):
    def problem(self):
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

    def random_data(self, n: int):
        v = np.random.randint(1, 10, n)
        w = np.random.randint(1, 10, n)
        C = np.random.randint(10, 20)
        return {"v": v, "w": w, "C": C}
