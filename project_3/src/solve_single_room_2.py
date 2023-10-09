import numpy as np
from scipy.sparse import diags, linalg
import matplotlib.pyplot as plt
from pprint import pprint
from dataclasses import dataclass


@dataclass
class Boundary:
    left_vals: np.ndarray
    top_vals: np.ndarray
    right_vals: np.ndarray
    bot_vals: np.ndarray
    left_dirichlet: np.ndarray
    top_dirichlet: np.ndarray
    right_dirichlet: np.ndarray
    bot_dirichlet: np.ndarray


def calculate_neuman_values(boundary: Boundary, temperature: np.ndarray, h):
    lef = np.zeros_like(boundary.left_dirichlet)
    for i, b in enumerate(boundary.left_dirichlet):
        if b == 0:
            lef[i] = h * (
                -3 * temperature[i, 0]
                + temperature[i + 1, 0]
                + temperature[i - 1, 0]
                + temperature[i, 1]
            )
    rig = np.zeros_like(boundary.right_dirichlet)
    for i, b in enumerate(boundary.right_dirichlet):
        if b == 0:
            rig[i] = h * (
                -3 * temperature[i, -1]
                + temperature[i + 1, -1]
                + temperature[i - 1, -1]
                + temperature[i, -2]
            )
    bot = np.zeros_like(boundary.bot_dirichlet)
    for i, b in enumerate(boundary.bot_dirichlet):
        if b == 0:
            bot[i] = h * (
                -3 * temperature[0, i]
                + temperature[0, i + 1]
                + temperature[0, i - 1]
                + temperature[1, i]
            )
    top = np.zeros_like(boundary.top_dirichlet)
    for i, b in enumerate(boundary.top_dirichlet):
        if b == 0:
            top[i] = h * (
                -3 * temperature[-1, i]
                + temperature[-1, i + 1]
                + temperature[-1, i - 1]
                + temperature[-2, i]
            )
    return lef, top, rig, bot


def solve_dirichlet(boundary: Boundary):
    x_N = len(boundary.bot_vals)
    y_N = len(boundary.left_vals)
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
        [np.flip(d2), np.flip(d1), d0, d1, d2], [-y_N, -1, 0, 1, y_N], format="csc"
    )
    # change elements with neumann conditions

    A = A.tocsc()
    # print(A.toarray())
    b = np.zeros(N)
    b[:y_N] = boundary.left_vals
    b[-y_N:] = boundary.right_vals
    b[::y_N] = boundary.bot_vals
    b[y_N - 1:: y_N] = boundary.top_vals

    return linalg.spsolve(A, b).reshape(y_N, x_N, order="F"), A, b


def solve_neumann(boundary: Boundary):
    x_N = len(boundary.bot_vals)
    y_N = len(boundary.left_vals)
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
        [np.flip(d2), np.flip(d1), d0, d1, d2], [-y_N, -1, 0, 1, y_N], format="lil"
    )

    b = np.zeros(N)
    b[:y_N] = boundary.left_vals
    b[-y_N:] = boundary.right_vals
    b[::y_N] = boundary.bot_vals
    b[y_N - 1:: y_N] = boundary.top_vals

    for i in range(y_N):
        l_idx = i
        r_idx = i - y_N
        if boundary.left_dirichlet[i] == 0:
            A[l_idx, l_idx] = -3
            A[l_idx, l_idx+1] = 1
            A[l_idx, l_idx-1] = 1
            A[l_idx, l_idx + y_N] = 1

        if boundary.right_dirichlet[i] == 0:
            A[r_idx, r_idx] = -3
            A[r_idx, r_idx+1] = 1
            A[r_idx, r_idx-1] = 1
            A[r_idx, r_idx - y_N] = 1
    A = A.tocsc()
    # print("="*50)
    # print(A.toarray())
    # print("="*50)

    # print(b)

    return linalg.spsolve(A, b).reshape(y_N, x_N, order="F"), A, b


def plot_temp(temps):
    for i, temp in enumerate(temps):
        plt.figure(i)

        y_N, x_N = temp.shape
        X, Y = np.meshgrid(
            np.linspace(0, 1, x_N), np.linspace(0, 1.0 * (y_N) / (x_N), y_N)
        )

        # cmap = clr.LinearSegmentedColormap.from_list('cmap for Dennis Hein',
        #                                              [(0, '#ff0000'), (70/100., '#ffff00'), (100/100., '#00ff00')], N=64)
        plt.title("Temperature in the room")
        plt.contourf(X, Y, temp, levels=500,
                     cmap=plt.cm.coolwarm, vmin=5, vmax=40)
        plt.colorbar()

    plt.show()


if __name__ == "__main__":
    wall = 15
    heater = 40
    window = 5

    # Omega 1
    x_N_13 = 10
    y_N_13 = x_N_13
    x_N_2 = x_N_13
    y_N_2 = 2 * x_N_2
    h = 1/(x_N_13-1)

    # masks of shared boundraies
    gamma_1_mask = np.zeros(y_N_13)
    gamma_1_mask[0] = 1
    gamma_1_mask[-1] = 1
    gamma_2_mask = gamma_1_mask
    print(gamma_1_mask)

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
    # temp, A, b = solve_dirichlet(x_N_13, y_N_13, omega_1)
    # pprint(omega_1)
    # calculate_neuman_values(omega_1, temp)
    # pprint(omega_1)
    # print(temp)

    # Omega 2

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
    omega_2.left_vals[omega_2.left_dirichlet == 0] = 0

    # Omega 3
    omega_3 = Boundary(
        wall * np.ones(y_N_13),
        wall * np.ones(x_N_13),
        heater * np.ones(y_N_13),
        wall * np.ones(x_N_13),
        gamma_2_mask,
        np.ones(x_N_13),
        np.ones(y_N_13),
        np.ones(x_N_13),
    )

    # exit()

    w = 0.8
    t1s = []
    t2s = []
    t3s = []
    for i in range(10):

        t2, A, b = solve_dirichlet(omega_2)
        # print(t2)
        lef, top, rig, bot = calculate_neuman_values(omega_2, t2, h)

        idx12 = omega_1.right_dirichlet == 0
        idx21 = omega_2.left_dirichlet == 0
        idx23 = omega_2.right_dirichlet == 0
        idx32 = omega_3.left_dirichlet == 0
        print(rig)
        omega_1.right_vals[idx12] = lef[idx21]
        omega_3.left_vals[idx32] = rig[idx23]

        t1, A, b = solve_neumann(omega_1)

        t3, _, _ = solve_neumann(omega_3)

        omega_2.left_vals[idx21] = t1[idx12, -1]
        omega_2.right_vals[idx23] = t3[idx32, 0]
        if i != 0 and False:
            t1 = w * t1 + (1-w) * t1s[-1]
            t2 = w * t2 + (1-w) * t2s[-1]
            t3 = w * t3 + (1-w) * t3s[-1]
        t1s.append(t1.copy())
        t2s.append(t2.copy())
        t3s.append(t3.copy())

    # print(t1s[:2])
    # print(t2)
    # t3s.append(t3)
    # print(t1)
    # print(t2)
    # pprint(omega_2)

    # temps = [temp1, temp2]
    # print(temp1)
    print("="*50)
    print(t1)
    print("="*50)
    print(t2)
    print("="*50)
    print(t3)
    plot_temp([t1, t2, t3])

    # X, Y = np.meshgrid(
    #     np.linspace(0, 1, x_N - 2), np.linspace(0, 1.0 * (y_N - 2) / (x_N - 2), y_N - 2)
    # )
    # plt.title("Temperature in the room")
    # plt.contourf(X, Y, temp[1:-1, 1:-1], levels=500, cmap=plt.cm.coolwarm)
    # plt.colorbar()
    # plt.show()
