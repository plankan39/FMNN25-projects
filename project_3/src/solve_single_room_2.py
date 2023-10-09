import numpy as np
from scipy.sparse import diags, linalg
import matplotlib.pyplot as plt
from pprint import pprint
from dataclasses import dataclass


@dataclass
class Boundary:
    left: np.ndarray
    top: np.ndarray
    right: np.ndarray
    bot: np.ndarray
    left_dirichlet: np.ndarray
    top_dirichlet: np.ndarray
    right_dirichlet: np.ndarray
    bot_dirichlet: np.ndarray


def solve_dirichlet(x_N: int, y_N: int, boundary: Boundary):
    A = construct_matrix(x_N, y_N)
    b = construct_boundary(boundary)
    return linalg.spsolve(A, b)


def construct_matrix(x_N: int, y_N: int):
    # Main diagonal
    d_0 = -4
    # The agacent diagonals under and over d_0
    d_1 = np.ones(x_N * y_N - 1)
    # Change the corner points where Boundary conditions are imposed.
    # for k in range(1, x_N):
    #     d_1[k * y_N - 1] = 0

    d_1[[k * y_N - 1 for k in range(1, x_N)]] = 0
    # The two last diagonals that are y_N
    d_y_N = 1

    return diags(
        [d_0, d_1, d_1, d_y_N, d_y_N],
        [0, 1, -1, y_N, -y_N],
        shape=(x_N * y_N, x_N * y_N),
        format="csc",
    )


def construct_boundary(boundary: Boundary):
    x_N = len(boundary.top)
    y_N = len(boundary.left)

    b = np.zeros(x_N * y_N)
    b[-y_N:] -= boundary.right
    b[:y_N] -= boundary.left
    b[::y_N] -= boundary.bot
    b[y_N - 1 :: y_N] -= boundary.top

    return b


def reconstruct_result_matrix_with_boundries(theta: np.ndarray, x_N: int, y_N):
    return theta.reshape(y_N, x_N, order="F")



def solve_dirichlet2(x_N: int, y_N: int, boundary: Boundary):
    # Total number of unknown points
    N = x_N * y_N

    d0 = np.ones(N)
    d1 = np.zeros(N - 1)
    d2 = np.zeros(N - y_N)

    for i in range(1, x_N - 1):
        d0[i * y_N + 1 : (i + 1) * y_N - 1] = -4
        d1[i * y_N + 1 : (i + 1) * y_N - 1] = 1
        d2[i * y_N + 1 : (i + 1) * y_N - 1] = 1

    A = diags([np.flip(d2), np.flip(d1), d0, d1, d2], [-y_N, -1, 0, 1, y_N],format="csc")
    b = np.zeros(N)
    b[:y_N] = boundary.left
    b[-y_N:] = boundary.right
    b[::y_N] = boundary.bot
    b[y_N - 1 :: y_N] = boundary.top


    return linalg.spsolve(A, b).reshape(y_N, x_N, order="F"), A, b

if __name__ == "__main__":
    # Omega 1
    
    # Number of discrete points along the x-axis
    x_N = 5
    y_N = 5

    bs = Boundary(
        15 * np.ones(x_N),
        40 * np.ones(y_N),
        15 * np.ones(x_N),
        5 * np.ones(y_N),
        np.ones(x_N),
        np.ones(y_N),
        np.ones(x_N),
        np.ones(y_N),
    )



    temp, A, b = solve_dirichlet2(x_N, y_N, bs)

    
    
    
    
    
    # Omega 2
    
    print(temp)

    # X, Y = np.meshgrid(np.linspace(0, 1, x_N-2), np.linspace(0, 1.0 * (y_N-2)/(x_N-2), y_N-2))
    # plt.title("Temperature in the room")
    # plt.contourf(X, Y, temp[1:-1,1:-1], levels=500, cmap=plt.cm.coolwarm)
    # plt.colorbar()
    # plt.show()

