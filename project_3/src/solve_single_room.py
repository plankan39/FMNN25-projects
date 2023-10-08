import numpy as np
from scipy.sparse import diags, linalg
import matplotlib.pyplot as plt
from pprint import pprint


def construct_neumann(unknown_boundry):
    pass


def construct_matrix(x_N: int, y_N: int):
    # Main diagonal
    d_0 = -4
    # The agacent diagonals under and over d_0
    d_1 = np.ones(x_N * y_N - 1)
    # Change the corner points where Boundary conditions are imposed.
    for k in range(1, x_N):
        d_1[k * y_N - 1] = 0

    d_1[[k * y_N - 1 for k in range(1, x_N)]] = 0
    # The two last diagonals that are y_N
    d_y_N = 1

    return diags(
        [d_0, d_1, d_1, d_y_N, d_y_N],
        [0, 1, -1, y_N, -y_N],
        shape=(x_N * y_N, x_N * y_N),
        format="csc",
    )


def generate_boundry(
    left_b: np.ndarray, right_b: np.ndarray, bot_b: np.ndarray, top_b: np.ndarray
):
    x_N = len(top_b)
    y_N = len(right_b)

    b = np.zeros(x_N * y_N)
    b[-y_N:] -= right_b
    b[:y_N] -= left_b
    b[::y_N] -= bot_b
    b[y_N - 1 :: y_N] -= top_b

    return b


def reconstruct_result_matrix_with_boundries(theta: np.ndarray, x_N: int, y_N):
    return theta.reshape(y_N, x_N, order="F")


if __name__ == "__main__":
    ################ Describing the problem and setting parameters ################

    # Describe the length of the sides of the rectangle
    x_len = 1.0
    y_len = 1.0
    # Number of discrete points along the x-axis
    x_N = 2

    # The stepsize for the discrete points
    h = x_len / x_N

    # Number of discrete points along y-axis
    y_N = int(y_len / h)

    # Total number of unknown points
    N = x_N * y_N

    # Boundaries
    top_b = 40 * np.ones(x_N)
    bot_b = 5 * np.ones(x_N)
    left_b = 15 * np.ones(y_N)
    right_b = 15 * np.ones(y_N)

    A = construct_matrix(x_N, y_N)

    b = generate_boundry(left_b, right_b, bot_b, top_b)
    theta = linalg.spsolve(A, b)

    T = reconstruct_result_matrix_with_boundries(theta, x_N, y_N)

    print(
        f"x_len = {x_len}, y_len = {y_len}, h = {h}, x_N = {x_N}, y_N = {y_N}, N = {N}"
    )
    print(f"A:\n{A.toarray()}")
    print(f"b:\n{b}")
    print(f"T:\n{T}")

    X, Y = np.meshgrid(np.linspace(0, x_len, x_N), np.linspace(0, y_len, y_N))
    plt.title("Temperature in the room")
    plt.contourf(X, Y, T, levels=500, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.show()
