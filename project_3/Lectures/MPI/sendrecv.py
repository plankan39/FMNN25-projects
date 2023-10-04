from mpi4py import MPI
import numpy as np
""" Get a communicator:
    The most common communicator is the
    one that connects all available processes
    which is called COMM_WORLD.
    Clone the communicator to avoid interference
    with other libraries or applications
"""
comm = MPI.Comm.Clone( MPI.COMM_WORLD )

rank = comm.Get_rank()
if rank == 0:
    # send 10 numbers to rank 1 (dest=1)
    data = np.array([range(1,10)])
    print("P[",rank,"] sent data =",data)
    # method 'send' for Python objects (pickle under the hood):
    comm.send(data, dest=1 )

if rank == 1:
    # receive 10 numbers from rank 0 (source=0)
    # method 'recv' for Python objects (pickle under the hood):
    data = comm.recv(source=0 )
    print("P[",rank,"] received data =",data)
