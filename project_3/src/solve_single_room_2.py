import numpy as np
from scipy.sparse import diags, linalg
import matplotlib.pyplot as plt
from pprint import pprint
from dataclasses import dataclass


@dataclass
class Boundary:
    left_neumann: np.ndarray
    top_neumann: np.ndarray
    right_neumann: np.ndarray
    bot_neumann: np.ndarray
    left_dirichlet: np.ndarray
    top_dirichlet: np.ndarray
    right_dirichlet: np.ndarray
    bot_dirichlet: np.ndarray


<< << << < HEAD


def calculate_neuman_values(boundary: Boundary, temperature: np.ndarray):
    left = np.zeros_like(boundary.left_dirichlet)
    for i, b in enumerate(boundary.left_dirichlet):
        if b == 0:
            left[i] = (
                -3 * temperature[i, 0]
                + temperature[i + 1, 0]
                + temperature[i - 1, 0]
                + temperature[i, 1]
            )

    right = np.zeros_like(boundary.right_dirichlet)
    for i, b in enumerate(boundary.right_dirichlet):
        if b == 0:
            right[i] = (
                -3 * temperature[i, -1]
                + temperature[i + 1, -1]
                + temperature[i - 1, -1]
                + temperature[i, -2]
            )

    bot = np.zeros_like(boundary.bot_dirichlet)
    for i, b in enumerate(boundary.bot_dirichlet):
        if b == 0:
            bot[i] = (
                -3 * temperature[0, i]
                + temperature[0, i + 1]
                + temperature[0, i - 1]
                + temperature[1, i]
            )

    top = np.zeros_like(boundary.top_dirichlet)
    for i, b in enumerate(boundary.top_dirichlet):
        if b == 0:
            top[i] = (
                -3 * temperature[-1, i]
                + temperature[-1, i + 1]
                + temperature[-1, i - 1]
                + temperature[-2, i]
            )

    return left, top, right, bot


def solve_dirichlet(x_N: int, y_N: int, boundary: Boundary):
    # Total number of unknown points
    N = x_N * y_N

    d0 = np.ones(N)
    d1 = np.zeros(N - 1)
    d2 = np.zeros(N - y_N)

    for i in range(1, x_N - 1):
        d0[i * y_N + 1: (i + 1) * y_N - 1] = -4
        d1[i * y_N + 1: (i + 1) * y_N - 1] = 1
        d2[i * y_N + 1: (i + 1) * y_N - 1] = 1

    # create matrix with dirichlet conditions
    A = diags(
        [np.flip(d2), np.flip(d1), d0, d1, d2], [-y_N, -1, 0, 1, y_N], format="lil"
    )
    # change elements with neumann conditions

    A[:, 0] = 1
    A = A.tocsc()
    # print(A.toarray())
    b = np.zeros(N)
    b[:y_N] = boundary.left
    b[-y_N:] = boundary.right
    b[::y_N] = boundary.bot
    b[y_N - 1:: y_N] = boundary.top

    print(b)

    return linalg.spsolve(A, b).reshape(y_N, x_N, order="F"), A, b


def solve_neumann(x_N: int, y_N: int, boundary: Boundary):
    # Total number of unknown points
    N = x_N * y_N

    d0 = np.ones(N)
    d1 = np.zeros(N - 1)
    d2 = np.zeros(N - y_N)

    for i in range(1, x_N - 1):
        d0[i * y_N + 1: (i + 1) * y_N - 1] = -4
        d1[i * y_N + 1: (i + 1) * y_N - 1] = 1
        d2[i * y_N + 1: (i + 1) * y_N - 1] = 1

    A = diags(
        [np.flip(d2), np.flip(d1), d0, d1, d2], [-y_N, -1, 0, 1, y_N], format="csc"
    )
    print(A.toarray())
    b = np.zeros(N)
    b[:y_N] = boundary.left
    b[-y_N:] = boundary.right
    b[::y_N] = boundary.bot
    b[y_N - 1:: y_N] = boundary.top

    print(b)

    return linalg.spsolve(A, b).reshape(y_N, x_N, order="F"), A, b


if __name__ == "__main__":
    wall = 15
    heater = 40
    window = 5

    # Omega 1
    x_N_13 = 4
    y_N_13 = x_N_13

# masks of shared boundraies
gamma_1_mask = np.concatenate((1, np.zeros(y_N_13 - 2), 1))
gamma_2_mask = gamma_1_mask

omega_1 = Boundary(
    heater * np.ones(y_N_13),  # left
    wall * np.ones(x_N_13),  # top
    wall * np.ones(y_N_13),  # right, shared with omega 2
    wall * np.ones(x_N_13),  # bottom
    np.ones(y_N_13),  # left,
    np.ones(x_N_13),  # top,
    gamma_1_mask,  # right, this is shared have some zeros
    np.ones(x_N_13),  # bottom
)
temp, A, b = solve_dirichlet(x_N_13, y_N_13, omega_1)
print(temp)
print(calculate_neuman_values(omega_1, temp))

# Omega 2
x_N_2 = x_N_13
y_N_2 = 2 * x_N_2

omega_2 = Boundary(
    wall * np.ones(y_N_2),  # left, partially shared with omega_1
    heater * np.ones(x_N_2),  # top
    wall * np.ones(y_N_2),  # right, partially shared with omega_3
    window * np.ones(x_N_2),  # bot
    np.concatenate((gamma_1_mask, np.ones(
        y_N_2 - len(gamma_1_mask)))),  # left
    np.ones(x_N_2),  # top
    np.concatenate(
        (np.ones(y_N_2 - len(gamma_2_mask)), gamma_2_mask)),  # right
    np.ones(x_N_2),  # bot
)

# Omega 3
omega_3 = Boundary(
    heater * np.ones(y_N_13),
    wall * np.ones(x_N_13),
    wall * np.ones(y_N_13),
    wall * np.ones(x_N_13),
    gamma_2_mask,
    np.ones(x_N_13),
    np.ones(y_N_13),
    np.ones(x_N_13),
)

X, Y = np.meshgrid(np.linspace(0, 1, x_N-2),
                   np.linspace(0, 1.0 * (y_N-2)/(x_N-2), y_N-2))
plt.title("Temperature in the room")
plt.contourf(X, Y, temp[1:-1, 1:-1], levels=500, cmap=plt.cm.coolwarm)
plt.colorbar()
plt.show()


# def solve_dirichlet(x_N: int, y_N: int, boundary: Boundary):
#     A = construct_matrix(x_N, y_N)
#     b = construct_boundary(boundary)
#     return linalg.spsolve(A, b)


# def construct_matrix(x_N: int, y_N: int):
#     # Main diagonal
#     d_0 = -4
#     # The agacent diagonals under and over d_0
#     d_1 = np.ones(x_N * y_N - 1)
#     # Change the corner points where Boundary conditions are imposed.
#     # for k in range(1, x_N):
#     #     d_1[k * y_N - 1] = 0

#     d_1[[k * y_N - 1 for k in range(1, x_N)]] = 0
#     # The two last diagonals that are y_N
#     d_y_N = 1

#     return diags(
#         [d_0, d_1, d_1, d_y_N, d_y_N],
#         [0, 1, -1, y_N, -y_N],
#         shape=(x_N * y_N, x_N * y_N),
#         format="csc",
#     )


# def construct_boundary(boundary: Boundary):
#     x_N = len(boundary.top)
#     y_N = len(boundary.left)

#     b = np.zeros(x_N * y_N)
#     b[-y_N:] -= boundary.right
#     b[:y_N] -= boundary.left
#     b[::y_N] -= boundary.bot
#     b[y_N - 1 :: y_N] -= boundary.top

#     return b


# def reconstruct_result_matrix_with_boundries(theta: np.ndarray, x_N: int, y_N):
#     return theta.reshape(y_N, x_N, order="F")
