import jijmodeling as jm
import numpy as np

from .problem import Problem


class CVRP(Problem):

    def _set_problem(self, name: str):
        Q = jm.Placeholder("Q", description="Capacity of vehicles")
        D = jm.Placeholder("D", ndim=1, description="Demand of customers")
        c = jm.Placeholder("c", ndim=2, description="Cost matrix")
        n = c.len_at(0, latex="n")

        x = jm.BinaryVar("x", shape=(n, n), description="use edge or not")
        i, j = jm.Element("i", n), jm.Element("j", n)

        jmC = jm.Constraint

        problem = jm.Problem(name)

        problem += jm.sum([i, j], c[i, j] * x[i, j])
        problem += jmC("no loop", x[i, i] == 0, forall=i)
        problem += jmC("in", x[i, :].sum() == 1, forall=[(i, i > 0)])
        problem += jmC("out", x[:, i].sum() == 1, forall=[(i, i > 0)])

        self._problem = problem
        self.Q = Q
        self.D = D
        self.c = c
        self.x = x
        self.ij = (i, j)
        self.n = n

    def random_data(self, n: int):
        x, y = np.random.rand(2, n)
        depo = (0.5, 0.5)
        x = np.concatenate([[depo[0]], x])
        y = np.concatenate([[depo[1]], y])
        distance = np.sqrt(
            (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
        )
        q = np.random.randint(1, 10, size=n + 1)

        # For feasible solution
        Q = q.sum() // 4 + q.max()
        return {
            "c": distance,
            "D": q,
            "Q": Q,
            "xy": np.array([x, y]),
        }


class CVRPMTZ(CVRP):
    def problem(self):

        self._set_problem("CVRPMTZ")

        u = jm.ContinuousVar(
            "u",
            shape=(self.n,),
            lower_bound=self.D,
            upper_bound=self.Q,
            description="cumulative demand",
        )

        jmC = jm.Constraint
        i, j = self.ij
        self._problem += jmC(
            "potiantial",
            u[i]
            + self.D[j]
            - self.Q * (1 - self.x[i, j])
            + (self.Q - self.D[i] - self.D[j]) * self.x[j, i]
            <= u[j],
            forall=[i, (j, j != 0)],
        )
        return self._problem


class CVRPFlow(CVRP):
    def problem(self):
        self._set_problem("CVRPFlow")
        x = jm.BinaryVar("x", shape=(self.n, self.n), description="use edge or not")
        f = jm.ContinuousVar(
            "f",
            shape=(self.n, self.n),
            lower_bound=0,
            upper_bound=self.Q,
            description="cumulative demand",
        )

        jmC = jm.Constraint
        i, j = self.ij
        self._problem += jmC(
            "flow",
            jm.sum(j, f[j, i]) - jm.sum(j, f[i, j]) == self.D[i],
            forall=[(i, i > 0)],
        )
        self._problem += jmC(
            "flow cap", f[i, j] <= (self.Q - self.D[i]) * x[i, j], forall=[i, j]
        )

        return self._problem
