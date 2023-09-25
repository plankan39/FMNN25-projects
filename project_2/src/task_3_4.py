import numpy as np
from scipy import optimize


class OptimizationProblem:
    def __init__(self, f, f_gradient=None):
        self.f = f
        self.f_gradient = f_gradient
# class Optimizer:


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
        return -self.Qinv @ self.q

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


def TEST_classical_newton():
    n = 50
    (Q, q) = positive_definite_quadratic_data(n)
    quadratic = f_quadratic(Q, q, n)
    def f(x): return quadratic.eval(x)
    x0 = np.random.rand(n)

    x_list, f_list = minimize_classical_newton(
        f, x0, epsilon=1e-6, max_iter=10)

    f_min = quadratic.analytic_minimum()
    x_min = quadratic.analytic_minimizer()
    f_min_approx = f_list[-1]
    x_min_approx = x_list[-1]

    print(f"f_min:        {f_min}")
    print(f"f_min_approx: {f_min_approx }")
    print(f"L2(x-x_approx1): {np.linalg.norm(x_min - x_min_approx)}")


def finite_difference_gradient_alt(f, p, epsilon=0.01):
    # p = np.array(p)
    gradient = np.zeros_like(p)

    for i in range(len(p)):
        p_shifted_front = p.copy()
        p_shifted_back = p.copy()

        # print("Before addition:", p_shifted_front)
        p_shifted_front[i] += epsilon
        # print("After addition:", p_shifted_front)

        p_shifted_back[i] -= epsilon

        gradient[i] = (f(*p_shifted_front) -
                       f(*p_shifted_back)) / (2 * epsilon)

    return gradient


def finite_difference_gradient(f, x, h=0.01):
    """
    based on formulas from (Abramowitz and Stegun 1972) in
    https://www.sfu.ca/sasdoc/sashtml/iml/chap11/sect8.htm
    """

    n = x.shape[0]

    gradient = np.zeros_like(x)
    E = np.eye(n)
    hi = h
    for i in range(n):
        ei = E[:, i]
        f1 = f(x + hi * ei)
        f2 = f(x - hi * ei)
        df = (f1 - f2) / (2 * hi)
        gradient[i] = df
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


def calc_residual(gk):
    return np.linalg.norm(gk)


def calc_cauchy_diff(x_k_1, x_k):
    return np.linalg.norm(x_k_1 - x_k)


def minimize_classical_newton(f, x0, epsilon, max_iter):
    x_list = [x0]
    f_list = [f(x0)]
    x_new = x0
    for i in range(max_iter):
        x = x_new
        Ginv = np.linalg.inv(finite_difference_hessian(x, f, h=0.1))
        g = finite_difference_gradient(f, x, epsilon)
        s = -Ginv @ g
        x_new = x + s
        f_new = f(x_new)

        x_list.append(x_new)
        f_list.append(f_new)

        residual = calc_residual(g)
        cauchy = calc_cauchy_diff(x_new, x)
        if residual < epsilon or cauchy < epsilon:
            # print(f"residual {residual}")
            # print(f"cauchy {cauchy }")
            break

            # x_new = HESSIAN, GRADIENT)
    return x_list, f_list


def parametrize_function(f, gradient, point):
    """
    Returns a function 'g' such that g(x) gives the value of the function 'f' at a point determined by a step 'x' along the 'gradient' from 'point'.
    """
    gradient = np.array(gradient)
    point = np.array(point)
    return lambda x: f(*(point + gradient * x))


def parametrize_function_line(f, x, direction):
    """
    Returns a function 'g' such that g(x) gives the value of the function 'f' at a point determined by a step 'gamma' along the 'direction' from 'point'.
    """
    return lambda gamma: f(x + gamma * direction)


def exact_line_search(f, ak, bk, epsilon, alpha=0.618033988749):
    """
    Performs the exact line search for a function 'f' using the golden section
    method.

    f: The function to be minimized.
    ak, bk: The initial interval.
    alpha: The golden ratio constant.
    epsilon: The tolerance.
    Returns the x value where the function 'f' has a minimum in the interval [ak, bk].
    """
    # Iteratively reduce the interval [ak, bk] until its width is less than epsilon
    while abs(bk - ak) > epsilon:
        # print(f"Interval: [{ak}, {bk}]")

        # Using golden section search
        sigmak = ak + (1 - alpha) * (bk - ak)
        ugmak = ak + alpha * (bk - ak)

        # Determine new interval of uncertainty based on function values at sigmak and ugmak
        # FIXME: here we have one function evaluation too much, can be optimized
        if f(sigmak) > f(ugmak):
            ak = sigmak
        else:
            bk = ugmak

    return (bk + ak) / 2


def minimize_newton_exact_line_search(f, x0, epsilon, max_iter, ak=0, bk=1e8, line_search_epsilon=1e-4):
    x_list = [x0]
    f_list = [f(x0)]
    x_new = x0
    for i in range(max_iter):
        x = x_new
        Ginv = np.linalg.inv(finite_difference_hessian(x, f, h=0.1))
        g = finite_difference_gradient(f, x, epsilon)
        s = -Ginv @ g
        f_line = parametrize_function_line(f, x, s)
        gamma_min = exact_line_search(f_line, ak, bk,  line_search_epsilon)
        x_new = x + gamma_min * s
        f_new = f(x_new)

        x_list.append(x_new)
        f_list.append(f_new)

        residual = calc_residual(g)
        cauchy = calc_cauchy_diff(x_new, x)
        if residual < epsilon or cauchy < epsilon:
            # print(f"residual {residual}")
            # print(f"cauchy {cauchy }")
            break

            # x_new = HESSIAN, GRADIENT)
    return x_list, f_list


if __name__ == "__main__":
    # TEST_quadratic_minimum()
    # TEST_hessian_on_quadratic()
    # TEST_classical_newton()

    n = 10
    (Q, q) = positive_definite_quadratic_data(n)
    quadratic = f_quadratic(Q, q, n)
    def f(x): return quadratic.eval(x)
    x0 = np.random.rand(n)

    x_list, f_list = minimize_newton_exact_line_search(
        f, x0, epsilon=1e-6, max_iter=10)

    f_min = quadratic.analytic_minimum()
    x_min = quadratic.analytic_minimizer()
    f_min_approx = f_list[-1]
    x_min_approx = x_list[-1]

    print(f"f_min:        {f_min}")
    print(f"f_min_approx: {f_list[-1]}")
    print(f"L2(x-x_approx1): {np.linalg.norm(x_min - x_min_approx)}")
