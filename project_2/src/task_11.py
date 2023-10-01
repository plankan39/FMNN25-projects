import numpy as np
from scipy import optimize as scipy_optimize
import line_search as LineSearch
from optimization import Problem
from optimization.quasi_newton._bfgs import BFGS
import chebyquad_problem


if __name__ == "__main__":

    # starting points
    X0 = [
        np.array([0.42951458, 0.26069987, 0.24245167, 0.78074889]),

        np.array([0.86122376, 0.57092373, 0.15233913, 0.42263092,
                 0.88483327, 0.90447402, 0.3613921, 0.87774089]),

        np.array([0.45970799, 0.74899486, 0.54816307, 0.25772841,
                  0.15908466, 0.61995692, 0.10240036, 0.85029983,
                  0.5542435, 0.47892635, 0.53911505]),
    ]
    N = [4, 8, 11]

    f = chebyquad_problem.chebyquad
    f_grad = chebyquad_problem.gradchebyquad

    # parameters for optimisation used by both scipy and ours
    g_tol = 1e-5
    x_tol = 0
    norm_ord = 2
    max_iter = 500

    print("Scipy and ours BFGS comparison chebyquad")
    print("="*50)
    for x0, n in zip(X0, N):

        # should converge after 18 iterations
        xmin_scipy = scipy_optimize.fmin_bfgs(
            f=f, x0=x0, fprime=f_grad, disp=0, gtol=g_tol, xrtol=x_tol, norm=2, maxiter=max_iter)

        problem = Problem(chebyquad_problem.chebyquad,
                          chebyquad_problem.gradchebyquad)

        # line_seach_scipy = LineSearch.PowellWolfeScipy(problem.objective_function,
        #                                                problem.gradient_function)
        line_seach = LineSearch.PowellWolfeBenja(problem.objective_function,
                                                 problem.gradient_function)
        optimization = BFGS(problem, line_seach,
                            max_iterations=max_iter, g_tol=g_tol, x_tol=x_tol)
        optimization.optimize(x0)
        xmin_ours = optimization.xmin

        print(f"n = {n}")
        print(f"    scipy bfgs f(x_min): {f(xmin_scipy)}")
        print(f"    our bfgs f(x_min):   {f(xmin_ours)}")
        print(f"    num_iters:   {len(optimization.points)}")

        # optimization.report()
