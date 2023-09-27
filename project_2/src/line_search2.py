from dataclasses import dataclass
from collections.abc import Callable
from typing import Protocol
import numpy as np


@dataclass
class OptimizationProblem:
    """Class describing an optimazation problem"""

    f: Callable[[np.ndarray], float]
    gradF: Callable[[np.ndarray], np.ndarray]
    hessF: Callable[[np.ndarray], np.ndarray]


class LineSearch(Protocol):
    """Protocol class for different implementation of line search"""

    problem: OptimizationProblem
    x: np.ndarray
    direction: np.ndarray

    def search(self, stepSizeInitial: float):
        """Perform line search to find the best step size alpha

        Args:
            stepSizeInitial (float): the initial stepsize guess

        Returns:
            float: the optimal step size alpha
        """
        ...


class PowellWolfe(LineSearch):
    def __init__(
        self,
        problem: OptimizationProblem,
        x: np.ndarray,
        direction: np.ndarray,
        rho: float,
        sigma: float,
    ) -> None:
        assert x.ndim == direction.ndim

        self.problem = problem
        self.x = x
        self.direction = direction
        self.rho = rho
        self.sigma = sigma

    def phi(self, alpha: float) -> float:
        """phi(alpha) = f(x + alpha * descent_direction)"""
        return self.problem.f(self.x + alpha * self.direction)

    def phi_prime(self, alpha: float) -> float:
        """The deravitive of phi"""
        return self.direction.dot(self.problem.gradF(self.x + alpha * self.direction))

    def armijo(self, alpha: float) -> bool:
        """Checks the armijo condition for a specific stepsize alpha"""
        return self.phi(alpha) <= self.phi(0) + self.sigma * alpha * self.phi_prime(0)

    def wolfe(self, alpha: float) -> bool:
        """Checks the second Powell-Wolfe condition"""
        print(f"phi_prim(alfa) {self.phi_prime(alpha)}")
        print(f"rho * phi_0 {self.rho * self.phi_prime(0)}")
        
        return self.phi_prime(alpha) >= self.rho * self.phi_prime(0)

    def search(self, stepSizeInitial: float) -> float:
        alpha_plus = stepSizeInitial
        alpha_minus = stepSizeInitial

        
        # Find lower and upper bound for alpha that fulfills armijo condition
        while not self.armijo(alpha_minus):
            alpha_minus /= 2
            print(f"alpha lower {alpha_minus}")

        while self.armijo(alpha_plus):
            alpha_plus *= 2
            print(f"alpha higher {alpha_plus}")

        # Find a value between the bounds that fulfills the second condition
        while not self.wolfe(alpha_minus):
            alpha_0 = (alpha_plus + alpha_minus) / 2
            print(alpha_0)
            if self.armijo(alpha_0):
                alpha_minus = alpha_0
            else:
                alpha_plus = alpha_0

        return alpha_minus

if __name__ == "__main__":
    p = OptimizationProblem(
        f=lambda x: 0.5 * x[0] ** 2 + 4.5 * x[1] ** 2,
        gradF=lambda x: np.array((x[0], 9 * x[1])),
        hessF=lambda x: x,
    )
#    p = OptimizationProblem(
 #       f=lambda x: 0.5 * x[0] ** 2,
  #      gradF=lambda x: np.array((x[0])),
   #     hessF=lambda x: x,
    #)
    
    x_0 =np.array([44, 12])
    d_0 = np.array([-1, -1])
    
    sigma = 0.01
    rho = 0.9
    

    ls = PowellWolfe(p, x_0, d_0, rho, sigma)
    from scipy.optimize import line_search 
    
    res = line_search(p.f, p.gradF, x_0, d_0, c1=sigma, c2=rho)
        
        
        

    a = ls.search(1)

        
    print(res)
    print(a)
