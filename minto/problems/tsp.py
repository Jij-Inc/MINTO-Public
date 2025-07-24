import jijmodeling as jm
import numpy as np

from .problem import Problem


class QuadTSP(Problem):
    def problem(self) -> jm.Problem:
        d = jm.Placeholder("d", ndim=2)
        n = d.len_at(0, latex="n")
        x = jm.BinaryVar("x", shape=(n, n))
        i, j, t = jm.Element("i", n), jm.Element("j", n), jm.Element("t", n)

        problem = jm.Problem("QuadTSP")
        jmC = jm.Constraint
        problem += jmC("one-city", x[i, :].sum() == 1, forall=i)
        problem += jmC("one-time", x[:, t].sum() == 1, forall=t)

        problem += jm.sum([i, j], d[i, j] * jm.sum(t, x[i, t] * x[j, (t + 1) % n]))

        return problem

    def random_data(self, n: int) -> dict:
        x, y = np.random.rand(2, n)
        d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
        return {"d": d, "x": x, "y": y}

    def data(self, n: int) -> dict:
        x, y = np.random.rand(2, n)
        d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
        return {"d": d, "x": x, "y": y}
