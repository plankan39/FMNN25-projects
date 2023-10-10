from mpi4py import MPI
import numpy as np
from room import Room, Boundary, plot_temp

# Tunable parameters
WALL = 15
HEATER = 40
WINDOW = 5
W = 0.8
ITERATIONS = 10

# Describing room 1
X_N_1 = 21
Y_N_1 = X_N_1
# Describing room 2
X_N_2 = X_N_1
Y_N_2 = 2 * X_N_2
# Describing room 3
X_N_3 = X_N_1
Y_N_3 = Y_N_1

# The
H = 1 / (X_N_1 - 1)


def gamma1():
    return Boundary(WALL * np.ones(Y_N_1), np.array([1] + [0] * (Y_N_1 - 2) + [1]))


def gamma2():
    return Boundary(WALL * np.ones(Y_N_3), np.array([1] + [0] * (Y_N_3 - 2) + [1]))


def omega1() -> Room:
    lb1 = Boundary(HEATER * np.ones(Y_N_1), np.ones(Y_N_1))
    tb1 = Boundary(WALL * np.ones(X_N_1), np.ones(X_N_1))
    rb1 = gamma1()
    bb1 = Boundary(WALL * np.ones(X_N_1), np.ones(X_N_1))
    return Room(lb1, tb1, rb1, bb1)


def omega2() -> Room:
    g1 = gamma1()
    g2 = gamma2()
    lb2 = g1 + Boundary(
        WALL * np.ones(Y_N_2 - g1.values.shape[0]),
        np.ones(Y_N_2 - g1.values.shape[0]),
    )
    tb2 = Boundary(HEATER * np.ones(X_N_2), np.ones(X_N_2))
    rb2 = Boundary(WALL * np.ones(Y_N_2 - len(g2)), np.ones(Y_N_2 - len(g2))) + g2
    bb2 = Boundary(WINDOW * np.ones(X_N_2), np.ones(X_N_2))
    return Room(lb2, tb2, rb2, bb2)


def omega3() -> Room:
    lb3 = gamma2()
    tb3 = Boundary(WALL * np.ones(X_N_3), np.ones(X_N_3))
    rb3 = Boundary(HEATER * np.ones(Y_N_3), np.ones(Y_N_3))
    bb3 = Boundary(WALL * np.ones(X_N_3), np.ones(X_N_3))
    return Room(lb3, tb3, rb3, bb3)


if __name__ == "__main__":
    comm = MPI.Comm.Clone(MPI.COMM_WORLD)

    rank = comm.Get_rank()
    temps = []

    if rank == 0:
        omega = omega1()
        for i in range(ITERATIONS):
            neumann: Boundary = comm.recv(source=1)
            omega.right.values[omega.right.dirichlet == 0] = neumann.values[
                neumann.dirichlet == 0
            ]
            temperature, *_ = omega.solveNeumann()
            if i != 0:
                temperature = W * temperature + (1-W) * temps[-1]
            temps.append(temperature)

            dirichlet = Boundary(temperature[:, -1], omega.right.dirichlet)
            comm.send(dirichlet, dest=1)
        comm.send(temps, 3)

    if rank == 1:
        omega = omega2()
        for i in range(ITERATIONS):
            temperature, *_ = omega.solveDirichlet()

            left, _, right, _ = omega.neumannValues(temperature, H)

            comm.send(left, dest=0)
            comm.send(right, dest=2)

            dirichlet_left: Boundary = comm.recv(source=0)
            dirichlet_right: Boundary = comm.recv(source=2)

            omega.left.values[omega.left.dirichlet == 0] = dirichlet_left.values[
                dirichlet_left.dirichlet == 0
            ]
            omega.right.values[omega.right.dirichlet == 0] = dirichlet_right.values[
                dirichlet_right.dirichlet == 0
            ]

            if i != 0:
                temperature = W * temperature + (1-W) * temps[-1]
            
            temps.append(temperature)
        comm.send(temps, 3)
    if rank == 2:
        for i in range(ITERATIONS):
            omega = omega3()
            neumann: Boundary = comm.recv(source=1)
            omega.left.values[omega.left.dirichlet == 0] = neumann.values[
                neumann.dirichlet == 0
            ]
            temperature, *_ = omega.solveNeumann()

            dirichlet = Boundary(temperature[:, 0], omega.left.dirichlet)
            if i != 0:
                temperature = W * temperature + (1-W) * temps[-1]
            temps.append(temperature)
            comm.send(dirichlet, dest=1)
        # print("\n", "#" * 40, f" {rank} ", "#" * 40, "\n", temperature, "\n", "#" * 88)
        comm.send(temps, 3)

    if rank == 3:
        t1 = comm.recv(source=0)
        t2 = comm.recv(source=1)
        t3 = comm.recv(source=2)

        np.set_printoptions(precision=1)
        print("\n", "=" * 38, " Ω1 ", "=" * 38)
        print(t1[-1])
        print("\n", "=" * 38, " Ω2 ", "=" * 38)
        print(t2[-1])
        print("\n", "=" * 38, " Ω3 ", "=" * 38)
        print(t3[-1])
        plot_temp([t1[-1], t2[-1], t3[-1]])
