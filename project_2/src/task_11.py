from scipy import optimize as scipy_optimize
from line_search import ExactLineSearch, PowellWolfeScipy
import numpy as np
from optimization import Problem
from optimization.quasi_newton._bfgs import BFGS
import chebyquad_problem


if __name__ == "__main__":

    f = chebyquad_problem.chebyquad
    f_grad = chebyquad_problem.gradchebyquad

    # parameters
    g_tol = 1e-5
    x_tol = 0
    norm_ord = 2
    max_iter = 500

    X0 = [
        np.array([0.42951458, 0.26069987, 0.24245167, 0.78074889]),

        np.array([0.86122376, 0.57092373, 0.15233913, 0.42263092,
                 0.88483327, 0.90447402, 0.3613921, 0.87774089]),

        np.array([0.45970799, 0.74899486, 0.54816307, 0.25772841,
                  0.15908466, 0.61995692, 0.10240036, 0.85029983,
                  0.5542435, 0.47892635, 0.53911505]),
    ]

    N = [4, 8, 11]
    print("Scipy and ours BFGS comparison chebyquad")
    print("="*50)
    for x0, n in zip(X0, N):

        # should converge after 18 iterations
        xmin_scipy = scipy_optimize.fmin_bfgs(
            f=f, x0=x0, fprime=f_grad, disp=0, gtol=g_tol, xrtol=x_tol, norm=2, maxiter=max_iter)

        problem = Problem(chebyquad_problem.chebyquad,
                          chebyquad_problem.gradchebyquad)

        line_seach = PowellWolfeScipy(problem.objective_function,
                                      problem.gradient_function)
        # ls = PowellWolfeScipy(problem.objective_function, problem.gradient_function)
        # ls = ExactLineSearch(problem.objective_function)
        optimization = BFGS(problem, line_seach,
                            max_iterations=max_iter, g_tol=g_tol, x_tol=x_tol)
        optimization.optimize(x0)
        xmin_ours = optimization.xmin

        print(f"n = {n}")
        print(f"    scipy bfgs f(x_min): {f(xmin_scipy)}")
        print(f"    our bfgs f(x_min):   {f(xmin_ours)}")
        # optimization.report()
