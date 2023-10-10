from dataclasses import dataclass
import numpy as np
from scipy.sparse import diags, linalg
import matplotlib.pyplot as plt
from pprint import pprint


@dataclass
class Boundary:
    values: np.ndarray
    dirichlet: np.ndarray

    def __add__(self, other):
        return Boundary(
            np.concatenate((self.values, other.values)),
            np.concatenate((self.dirichlet, other.dirichlet)),
        )

    def __len__(self):
        return self.values.shape[0]


@dataclass
class Room:
    left: Boundary
    top: Boundary
    right: Boundary
    bottom: Boundary

    @property
    def xN(self):
        return self.top.values.shape[0]

    @property
    def yN(self):
        return self.left.values.shape[0]

    @property
    def N(self):
        return self.xN * self.yN

    def neumannValues(self, temperature: np.ndarray, h):
        left = Boundary(np.zeros(len(self.left)), self.left.dirichlet)
        right = Boundary(np.zeros(len(self.right)), self.right.dirichlet)
        for i, (l, r) in enumerate(zip(self.left.dirichlet, self.right.dirichlet)):
            if l == 0:
                left.values[i] = h * (
                    -3 * temperature[i, 0]
                    + temperature[i + 1, 0]
                    + temperature[i - 1, 0]
                    + temperature[i, 1]
                )
            if r == 0:
                right.values[i] = h * (
                    -3 * temperature[i, -1]
                    + temperature[i + 1, -1]
                    + temperature[i - 1, -1]
                    + temperature[i, -2]
                )

        top = Boundary(np.zeros(len(self.top)), self.top.dirichlet)
        bottom = Boundary(np.zeros(len(self.bottom)), self.bottom.dirichlet)
        for i, (t, b) in enumerate(zip(self.top.dirichlet, self.bottom.dirichlet)):
            if t == 0:
                top.values[i] = h * (
                    -3 * temperature[-1, i]
                    + temperature[-1, i + 1]
                    + temperature[-1, i - 1]
                    + temperature[-2, i]
                )
            if b == 0:
                bottom.values[i] = h * (
                    -3 * temperature[0, i]
                    + temperature[0, i + 1]
                    + temperature[0, i - 1]
                    + temperature[1, i]
                )
        return left, top, right, bottom

    def _getSystem(self, format="csc"):
        d0 = np.ones(self.N)
        d1 = np.zeros(self.N - 1)
        d2 = np.zeros(self.N - self.yN)

        for i in range(1, self.xN - 1):
            d0[i * self.yN + 1: (i + 1) * self.yN - 1] = -4
            d1[i * self.yN + 1: (i + 1) * self.yN - 1] = 1
            d2[i * self.yN + 1: (i + 1) * self.yN - 1] = 1

        b = np.zeros(self.N)
        b[: self.yN] = self.left.values
        b[-self.yN:] = self.right.values
        b[:: self.yN] = self.bottom.values
        b[self.yN - 1:: self.yN] = self.top.values

        A = diags(
            [np.flip(d2), np.flip(d1), d0, d1, d2],
            [-self.yN, -1, 0, 1, self.yN],
            format=format,
        )

        return A, b

    def solveNeumann(self):
        A, b = self._getSystem("lil")
        for i in range(self.yN):
            l_idx = i
            r_idx = i - self.yN
            if self.left.dirichlet[i] == 0:
                A[l_idx, l_idx] = -3
                A[l_idx, l_idx + 1] = 1
                A[l_idx, l_idx - 1] = 1
                A[l_idx, l_idx + self.yN] = 1

            if self.right.dirichlet[i] == 0:
                A[r_idx, r_idx] = -3
                A[r_idx, r_idx + 1] = 1
                A[r_idx, r_idx - 1] = 1
                A[r_idx, r_idx - self.yN] = 1

        # TODO: implement for top and bottom

        return linalg.spsolve(A.tocsc(), b).reshape(self.yN, self.xN, order="F"), A, b

    def solveDirichlet(self):
        A, b = self._getSystem()
        return linalg.spsolve(A, b).reshape(self.yN, self.xN, order="F"), A, b


def plot_temp(temps):
    for i, temp in enumerate(temps):
        plt.figure(i)
        y_N, x_N = temp.shape
        X, Y = np.meshgrid(
            np.linspace(0, 1, x_N), np.linspace(0, 1.0 * (y_N) / (x_N), y_N)
        )
        plt.title("Temperature in the room")
        plt.contourf(X, Y, temp, levels=500,
                     cmap=plt.cm.coolwarm, vmin=5, vmax=40)
        plt.colorbar()

    plt.show()


def plot_temp_rooms(temps):
    for i, T in enumerate(temps):
        t1, t2, t3 = T
        plt.figure(i)
        xn = 3*t2.shape[0]
        yn = t2.shape[1]
        t = np.zeros((xn, yn))
        t[0:xn, 0:yn//2] = t1
        t[xn:2*xn+1, 0:yn] = t2
        t[2*xn:2*xn+1, yn//2:yn] = t3
        X, Y = np.meshgrid(
            np.linspace(0, 1, x_N), np.linspace(0, 1.0 * (yn) / (xn), yn)
        )
        plt.title("Temperature in the room")
        plt.contourf(X, Y, t, levels=500,
                     cmap=plt.cm.coolwarm, vmin=5, vmax=40)
        plt.colorbar()

    plt.show()


# if __name__ == "__main__":
#     ##################################### Omega 1 #####################################
#     lb1 = Boundary(HEATER * np.ones(Y_N_1), np.ones(Y_N_1))
#     tb1 = Boundary(WALL * np.ones(X_N_1), np.ones(X_N_1))
#     bb1 = Boundary(WALL * np.ones(X_N_1), np.ones(X_N_1))
#     gamma1 = Boundary(WALL * np.ones(Y_N_1), np.array([1] + [0] * (Y_N_1 - 2) + [1]))
#     omega1 = Room(lb1, tb1, gamma1, bb1)

#     ##################################### Omega 3 #####################################
#     gamma2 = Boundary(WALL * np.ones(Y_N_3), np.array([1] + [0] * (Y_N_3 - 2) + [1]))
#     tb3 = Boundary(WALL * np.ones(X_N_3), np.ones(X_N_3))
#     rb3 = Boundary(HEATER * np.ones(Y_N_3), np.ones(Y_N_3))
#     bb3 = Boundary(WALL * np.ones(X_N_3), np.ones(X_N_3))
#     omega3 = Room(gamma2, tb3, rb3, bb3)

#     ##################################### Omega 2 #####################################
#     lb2 = gamma1 + Boundary(
#         WALL * np.ones(Y_N_2 - gamma1.values.shape[0]),
#         np.ones(Y_N_2 - gamma1.values.shape[0]),
#     )
#     tb2 = Boundary(HEATER * np.ones(X_N_2), np.ones(X_N_2))
#     rb2 = (
#         Boundary(WALL * np.ones(Y_N_2 - len(gamma2)), np.ones(Y_N_2 - len(gamma2)))
#         + gamma2
#     )
#     bb2 = Boundary(WINDOW * np.ones(X_N_2), np.ones(X_N_2))
#     omega2 = Room(lb2, tb2, rb2, bb2)

#     ########################### Dirchlet Neumann iterations ###########################
#     t1s = []
#     t2s = []
#     t3s = []
#     for i in range(ITERATIONS):
#         t2, *_ = omega2.solveDirichlet()
#         left, _, right, _ = omega2.neumannValues(t2, H)

#         idx12 = omega1.right.dirichlet == 0
#         idx21 = omega2.left.dirichlet == 0
#         idx23 = omega2.right.dirichlet == 0
#         idx32 = omega3.left.dirichlet == 0

#         omega1.right.values[idx12] = left.values[idx21]
#         t1, *_ = omega1.solveNeumann()
#         omega2.left.values[idx21] = t1[idx12, -1]

#         omega3.left.values[idx32] = right.values[idx23]
#         t3, *_ = omega3.solveNeumann()
#         omega2.right.values[idx23] = t3[idx32, 0]

#         if i != 0:
#             t1 = W * t1 + (1 - W) * t1s[-1]
#             t2 = W * t2 + (1 - W) * t2s[-1]
#             t3 = W * t3 + (1 - W) * t3s[-1]
#         t1s.append(t1)
#         t2s.append(t2)
#         t3s.append(t3)

#     np.set_printoptions(precision=0)
#     print("\n", "=" * 38, " Ω1 ", "=" * 38)
#     print(t1)
#     print("\n", "=" * 38, " Ω2 ", "=" * 38)
#     print(t2)
#     print("\n", "=" * 38, " Ω3 ", "=" * 38)
#     print(t3)
#     # plot_temp([t1, t2, t3])
