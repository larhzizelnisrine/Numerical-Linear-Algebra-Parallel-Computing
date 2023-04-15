import numpy as np
from scipy.sparse import lil_matrix
from numpy.random import rand, seed
from numba import njit
from mpi4py import MPI


# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Set seed for random number generation
seed(42)

# Define function for matrix-vector multiplication
@njit
def matrix_vector_mult(A, b, x):
    row, col = A.shape
    for i in range(row):
        a = A[i]
        for j in range(col):
            x[i] += a[j] * b[j]

    return x

# Set matrix and vector size
value = 1000

# Determine block size for each process
val = value // size

# Initialize matrix and vector for each process
if rank == 0:
    A = lil_matrix((value, value))
    A[0, :100] = rand(100)
    A[1, 100:200] = A[0, :100]
    A.setdiag(rand(value))
    A = A.toarray()
    b = rand(value)
else :
    A = None
    b = None

# Scatter matrix to each process
matrix = np.zeros((val, value))
comm.Scatter(A, matrix, root=0)

# Broadcast vector to each process
b = comm.bcast(b, root=0)

# Initialize result vector for each process
valx = np.zeros(val)

# Compute matrix-vector multiplication
start = MPI.Wtime()
matrix_vector_mult(matrix, b, valx)
stop = MPI.Wtime()

# Gather block sizes from each process to the root process
sendcounts = np.array(comm.gather(len(valx), root=0))

# Initialize result vector on root process
if rank == 0: 
    X = np.zeros(sum(sendcounts), dtype=np.double)
else :
    X = None

# Gather computed values from each process to the root process
comm.Gatherv(valx, recvbuf=(X, sendcounts, MPI.DOUBLE), root=0)

# Compare with result from dot product on root process
if rank == 0 :
    X_ = A.dot(b)
    print("CPU time of matrix multiplication is ", (stop - start) * 1000)
    print("The error comparing to the dot product is :", np.max(np.abs(X_ - X)))

