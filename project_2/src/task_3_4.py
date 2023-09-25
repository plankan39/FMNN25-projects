import numpy as np
from scipy import optimize


class f_quadratic:
    # defines a quadratic of the form
    # f(x) = 1/2 * xT*Q*x + qTx
    def __init__(self, Q, q, n):
        self.n = n
        assert (Q.shape == (n, n))
        assert (q.shape == (n,))
        self.Q = Q
        self.q = q
        self.qT = q.T
        self.Qinv = np.linalg.inv(Q)

    def eval(self, x):
        assert (x.shape == (self.n,))
        return 1/2 * x.T @ self.Q @ x + self.qT @ x

    def analytic_minimizer(self):
        return self.Qinv @ self.q

    def analytic_minimum(self):
        x_min = self.analytic_minimizer()
        return self.eval(x_min)


def positive_definite_quadratic_data(n, seed=True):
    """
    generates a positive definite Q and a random vector q

    :return: (Q,q)
    """
    if seed == False:
        rs = np.random
    else:
        rs = np.random.RandomState(
            np.random.MT19937(np.random.SeedSequence(seed)))
    Q = rs.randn(n, n)
    Q = Q.T@Q

    assert (np.all(np.linalg.eigvals(Q) > 0))
    q = rs.randn(n)

    return Q, q


def TEST_quadratic_minimum():
    # just quick test to check if the analytic minima of the quadratic function is
    # correct, seems to be right since it is very close to value returned from
    # scipy.optimize
    n = 10
    (Q, q) = positive_definite_quadratic_data(n)

    f = f_quadratic(Q, q, n)
    x = np.random.rand(n)

    res = optimize.minimize(f.eval, x, method='BFGS', tol=1e-9)
    x_scipy = res.x
    x_analytic = f.analytic_minimizer()
    print(x_analytic)
    print(x_scipy)
    print("diff between analytic solution and scipys bfgs solver solution",
          np.linalg.norm(x_scipy - x_analytic))


def finite_difference_gradient(x, f, h):
    """
    approximates the Hessian of f at x using finite differences
    """
    ...


def finite_difference_hessian(x, f, h):
    # based on formulas from (Abramowitz and Stegun 1972) in
    # https://www.sfu.ca/sasdoc/sashtml/iml/chap11/sect8.htm
    """
    approximates the Hessian of f at x using finite differences
    """
    n = x.shape[0]

    hessian = np.zeros((n, n))

    E = np.eye(n)
    hi = h
    hj = h
    for i in range(n):
        ei = E[:, i]
        # since the hessian is symetric we only evaluate upper triangle
        for j in range(i, n):
            if i == j:
                f1 = f(x + 2 * hi * ei)
                f2 = f(x + hi * ei)
                f3 = f(x)
                f4 = f(x - hi * ei)
                f5 = f(x - 2 * hi * ei)
                df = (-f1 + 16*f2 - 30*f3 + 16*f4 - f5) / (12 * hi * hi)
            else:
                ej = E[:, j]
                f1 = f(x + hi * ei + hj * ej)
                f2 = f(x + hi * ei - hj * ej)
                f3 = f(x - hi * ei + hj * ej)
                f4 = f(x - hi * ei - hj * ej)
                df = (f1 - f2 - f3 + f4) / (4 * hi * hj)
            hessian[i, j] = df
    return hessian


if __name__ == "__main__":
    # TEST_quadratic_minimum()
    n = 3
    (Q, q) = positive_definite_quadratic_data(n)
    f = f_quadratic(Q, q, n)
    x = np.random.rand(n)

    h = 0.1
    G = finite_difference_hessian(x, f.eval, h)
    print(G - G.T)
    print(np.linalg.norm(Q - G, ord=2))
