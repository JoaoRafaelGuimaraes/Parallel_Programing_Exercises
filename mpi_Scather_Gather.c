#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 10

int main(int argc, char** argv){


    int *fullbuf;
    int recbuff[SIZE];
    int world_size, world_rank, i, j;


    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    //Alocar espaço no heap para fullbuf -> apenas no host
    
    if (world_rank == 0){

        fullbuf = (int *)malloc(world_size * SIZE * sizeof(int)); //Cria vetor/matriz - Cada processo recebera 1/world_size
        //Popula fullbuf

        for(i = 0; i < world_size;i++)
            for (j=0;j<SIZE;j++)
                fullbuf[i*SIZE + j] = j;
                
        
    }//TODOS OS PROCESSOS DEVEM EXECUTAR O SCATTER:
    // int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
	//                 void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
	//                 MPI_Comm comm);
    MPI_Scatter(fullbuf, SIZE, MPI_INT, recbuff,SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Processo : %d\n",world_rank);
    for (i=0; i<SIZE;i++){
        recbuff[i] = recbuff[i]*world_rank;
        printf("%d ", recbuff[i]);

    }
    printf("\n");
    //RECEBE A INFORMAÇÃO PROCESSADA DE CADA UM DOS PROCESSOS
    // int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
	//                void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
	//                MPI_Comm comm);
    
    // MPI_Reduce (&sendbuf,&recvbuf,count,datatype,op,root,comm)

    int maximo[SIZE];
    MPI_Reduce(recbuff, &maximo, SIZE, MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

    
    if (world_rank == 0){
        printf("\nResultado do reduce:\n");
        for (i=0;i<SIZE;i++)
            printf("%d ", maximo[i]);
    }
    

    MPI_Finalize();
    return 0;
}