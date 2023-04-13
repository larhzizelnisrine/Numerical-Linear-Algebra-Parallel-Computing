

from mpi4py import MPI

# Initialize MPI and get the rank and size of the current process
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Qst 1 and 2                                                               
print("Hello World! I'm rank", rank);
if rank==0:
    print("Total no. of ranks =", size);

# Qst 3
if rank==0 :
    print("I have the right to print the rank", rank);
