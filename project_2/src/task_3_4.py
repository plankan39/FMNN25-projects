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


def TEST_hessian_on_quadratic():
    " quick test on quadratic function to see if Hessian estimation is accurate"
    n = 10
    (Q, q) = positive_definite_quadratic_data(n, seed=False)
    f = f_quadratic(Q, q, n)
    x = np.random.rand(n)

    h = 0.1
    G = Q
    G_approx = finite_difference_hessian(x, f.eval, h)
    print("L2 of difference between true hessian and approximated hessian of quadratic: \n",
          np.linalg.norm(G - G_approx, ord=2))


def finite_difference_gradient(f, p, epsilon=0.01):
    p = np.array(p)
    gradient = np.zeros_like(p)

    for i in range(len(p)):
        p_shifted_front = p.copy()
        p_shifted_back = p.copy()

        print("Before addition:", p_shifted_front)
        p_shifted_front[i] += epsilon
        print("After addition:", p_shifted_front)

        p_shifted_back[i] -= epsilon

        gradient[i] = (f(*p_shifted_front) -
                       f(*p_shifted_back)) / (2 * epsilon)

    return gradient


def finite_difference_hessian(x, f, h=0.1):
    """
    h=0.1 actually appears to give better approximations than smaller h
    when tested on quadratics
    based on formulas from (Abramowitz and Stegun 1972) in
    https://www.sfu.ca/sasdoc/sashtml/iml/chap11/sect8.htm

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
                hessian[i, j] = df
            else:
                ej = E[:, j]
                f1 = f(x + hi * ei + hj * ej)
                f2 = f(x + hi * ei - hj * ej)
                f3 = f(x - hi * ei + hj * ej)
                f4 = f(x - hi * ei - hj * ej)
                df = (f1 - f2 - f3 + f4) / (4 * hi * hj)
                hessian[i, j] = df
                hessian[j, i] = df
    return hessian


def stop_criterion_residual_fulfilled(fk, epsilon):
    """
    fk: f(x_k)
    """
    return np.linalg.norm(fk) < epsilon


def stop_criterion_cauchy_fulfilled(x_k_1, x_k, epsilon):
    return np.linalg.norm(x_k_1 - x_k) < epsilon


def minimize_classical_newton(f, x0, epsilon, max_iter):
    x_list = [x0]
    f_list = [f(x0)]
    x_new = x0
    for i in range(max_iter):
        x = x_new
        Ginv = np.linalg.inv(finite_difference_hessian(x, f, h=0.1))
        s = -Ginv @ x
        x_new = x + s
        f_new = f(x_new)

        x_list.append(x_new)
        f_list.append(f_new)

        stop = stop_criterion_residual_fulfilled(
            f_new, epsilon) or stop_criterion_cauchy_fulfilled(x_new, x, epsilon)

        if stop:
            break
        # x_new = HESSIAN, GRADIENT)
    return x_list, f_list


if __name__ == "__main__":
    # TEST_quadratic_minimum()
    # TEST_hessian_on_quadratic()

    n = 3
    (Q, q) = positive_definite_quadratic_data(n)
    quadratic = f_quadratic(Q, q, n)
    def f(x): return quadratic.eval(x)
    x0 = np.random.rand(n)

    x_list, f_list = minimize_classical_newton(
        f, x0, epsilon=1e-6, max_iter=10)

    print(f_list)
    print(len(x_list))
