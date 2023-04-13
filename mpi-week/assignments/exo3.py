from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

if RANK == 0:
    data = int(input('input a value : '))
else:
    data = COMM.recv(source=RANK-1)
    print("Process {} got data {} from process {}".format(RANK,data,RANK-1))

COMM.send(data, dest=(RANK+1)%SIZE)

if RANK == 0:
    data = COMM.recv(source=SIZE-1)
    print("Process {} got data {} from process {}".format(RANK,data,SIZE-1))
MPI.Finalize()

