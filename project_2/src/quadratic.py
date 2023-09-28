import numpy as np
from scipy import optimize


class f_quadratic:
    # defines a quadratic of the form
    # f(x) = 1/2 * xT*Q*x + qTx
    def __init__(self, Q, q, n):
        self.n = n
        assert Q.shape == (n, n)
        assert q.shape == (n,)
        self.Q = Q
        self.q = q
        self.qT = q.T
        self.Qinv = np.linalg.inv(Q)

    def eval(self, x):
        # assert (x.shape == (self.n,))
        return 1 / 2 * x.T @ self.Q @ x + self.qT @ x

    def analytic_minimizer(self):
        return -self.Qinv @ self.q

    def analytic_minimum(self):
        x_min = self.analytic_minimizer()
        return self.eval(x_min)


def positive_definite_quadratic_data(n, seed=True):
    """
    generates a positive definite Q and a random vector q,
    has analytic solution: https://en.wikipedia.org/wiki/Definite_quadratic_form

    :return: (Q,q)
    """
    if seed == False:
        rs = np.random
    else:
        rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))
    Q = rs.randn(n, n)
    Q = Q.T @ Q

    assert np.all(np.linalg.eigvals(Q) > 0)
    q = rs.randn(n)

    return Q, q


def TEST_quadratic_minimum():
    # just quick test to check if the analytic minima of the quadratic function is
    # correct, seems to be right since it is very close to value returned from
    # scipy.optimize
    n = 10
    (Q, q) = positive_definite_quadratic_data(n)

    f = f_quadratic(Q, q, n)
    x0 = np.random.rand(n)

    res = optimize.minimize(f.eval, x0, method="BFGS", tol=1e-9)
    x_scipy = res.x
    x_analytic = f.analytic_minimizer()
    print(x_analytic)
    print(x_scipy)
    print(
        "diff between analytic solution and scipys bfgs solver solution",
        np.linalg.norm(x_scipy - x_analytic),
    )
