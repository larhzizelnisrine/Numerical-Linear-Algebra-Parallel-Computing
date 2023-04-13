from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

while True:
    
    if RANK == 0:
        value = int(input('input a value: '))
    else:
        value = None 
    value = COMM.bcast(value,root=0)

    if value <0:
        break

    print('processor {} got {}'.format(RANK,value))
    COMM.Barrier()

MPI.Finalize()


