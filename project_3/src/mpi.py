from mpi4py import MPI
import numpy as np
from config import ITERATIONS, H, W, omega1, omega2, omega3
from room import Boundary, plot_temp_rooms

# from config import *


if __name__ == "__main__":
    comm = MPI.Comm.Clone(MPI.COMM_WORLD)

    rank = comm.Get_rank()
    temps = []

    if rank == 0:
        """Omega 1"""
        omega = omega1()
        for i in range(ITERATIONS):
            neumann: Boundary = comm.recv(source=1)
            omega.right.values[omega.right.dirichlet == 0] = neumann.values[
                neumann.dirichlet == 0
            ]
            temperature, A1, b = omega.solveNeumann()
            if i != 0:
                temperature = W * temperature + (1 - W) * temps[-1][0]
            temps.append((temperature, A1, b))

            dirichlet = Boundary(temperature[:, -1], omega.right.dirichlet)
            comm.send(dirichlet, dest=1)
        comm.send(temps, 3)

    if rank == 1:
        """Omega 2"""
        omega = omega2()
        for i in range(ITERATIONS):
            temperature, A2, b = omega.solveDirichlet()

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
                temperature = W * temperature + (1 - W) * temps[-1][0]

            temps.append((temperature, A2, b))
        comm.send(temps, 3)
    if rank == 2:
        """Omega 3"""
        for i in range(ITERATIONS):
            omega = omega3()
            neumann: Boundary = comm.recv(source=1)
            omega.left.values[omega.left.dirichlet == 0] = neumann.values[
                neumann.dirichlet == 0
            ]
            temperature, A3, b = omega.solveNeumann()

            dirichlet = Boundary(temperature[:, 0], omega.left.dirichlet)
            if i != 0:
                temperature = W * temperature + (1 - W) * temps[-1][0]
            temps.append((temperature, A3, b))
            comm.send(dirichlet, dest=1)
        comm.send(temps, 3)

    if rank == 3:
        """Presents the result"""
        o1 = comm.recv(source=0)
        o2 = comm.recv(source=1)
        o3 = comm.recv(source=2)

        np.set_printoptions(precision=1, threshold=10000, linewidth=200)
        for i in range(3):
            print("\n", "=" * 38, f" {i + 1} ", "=" * 38)
            t1, A1, b1 = o1[-i - 1]
            t2, A2, b2 = o2[-i - 1]
            t3, A3, b3 = o3[-i - 1]

            print(f"Ω1:\n{A1.toarray()}")
            print(f"Ω2:\n{A2.toarray()}")
            print(f"Ω3:\n{A3.toarray()}")

        print("\n", "=" * 38, " Ω1 ", "=" * 38)
        print(o1[-1][0])
        print("\n", "=" * 38, " Ω2 ", "=" * 38)
        print(o2[-1][0])
        print("\n", "=" * 38, " Ω3 ", "=" * 38)
        print(o3[-1][0])
        plot_temp_rooms([[o1[-1][0], o2[-1][0], o3[-1][0]]])
