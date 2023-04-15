from mpi4py import MPI
import numpy as np

N = 840

# initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# compute range of i for this process
start = rank * N // size
end = (rank + 1) * N // size

# compute local sum
sum_local = 0.0
for i in range(start, end):
    x = (i + 0.5) / N
    sum_local += 4.0 / (1.0 + x**2)

# reduce local sum to get global sum
sum_global = comm.reduce(sum_local, op=MPI.SUM, root=0)

# compute and print pi on root process
if rank == 0:
    pi = sum_global / N
    print(f"pi = {pi}")
 
#part 2
from mpi4py import MPI
import math

comm = MPI.COMM_WORLD
p = comm.Get_size()
rank = comm.Get_rank()

N = 840
BLOCKSIZE = N // p

partial_sum = 0.0
for i in range(rank*BLOCKSIZE, (rank+1)*BLOCKSIZE):
    partial_sum += 4.0 * math.pow(-1, i) / (2*i + 1)

print(f"Process {rank} computed a partial sum of {partial_sum}")

if rank == 0:
    total_sum = partial_sum
    for i in range(1, p):
        partial_sum = comm.recv(source=i)
        total_sum += partial_sum
    print(f"The total sum is {total_sum}")
else:
    comm.send(partial_sum, dest=0)

MPI.Finalize()    

