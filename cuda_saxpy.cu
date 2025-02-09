
#include <stdio.h>
// #implementando um saxpy
// #x e y são iniciados como x = 1.0f e y = 2.0f
// saxpy = y = ax + y

__global__
void saxpy(int n, float a, float *x, float *y){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<n)//garante que a posição do vetor existe
        y[i] = a*x[i] + y[i];
}


int main(){


int N = 1<<20;
float *x, *y, *d_x, *d_y;


//Aloca espaço na CPU
x = (float *)malloc(N*sizeof(float));   
y = (float *)malloc(N*sizeof(float));

// Carrega valores nos vetores

for (int i = 0; i<N;i++){

    x[i] = 1.0f;
    y[i] = 2.0f;
}

cudaMalloc(&d_x, N*sizeof(float) );
cudaMalloc(&d_y, N*sizeof(float) );

//Copia valores da CPU para A GPU
// cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )

cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);


//chama kernel
//Temos mais de 1 milhão de dados. Cada bloco terá o maior número possível de threads
int thread_per_block = 1024;
int num_blocks = (N + thread_per_block -1)/thread_per_block;
saxpy<<<num_blocks, thread_per_block>>>(N,2.0f,d_x,d_y);

cudaError_t err = cudaGetLastError();
if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));

//Copia o vetor y para a CPU novamente, para comparar e gerar o valor de max_error

cudaMemcpy(y,d_y,N*sizeof(float),cudaMemcpyDeviceToHost);

float max_error = 0.0f;
for (int i =0;i<N; i++){

    max_error = max(max_error, y[i] - 4.0f);
}

printf("Max Error = %f\n", max_error);

cudaFree(d_y);
cudaFree(d_x);
free(x);
free(y);

}