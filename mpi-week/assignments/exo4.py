from mpi4py import MPI
import numpy as np

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

n, m = 3, 6 

if RANK == 0:
    matrix = np.random.rand(n,m)
else:
    matrix = None

loc = np.empty((n//2, m//2), dtype=int)

COMM.Scatter(matrix, loc, root=0)

print(f"Process {RANK} received:\n{loc}")
MPI.Finalize()

