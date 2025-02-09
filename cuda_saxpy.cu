
#include <stdio.h>
// #implementando um saxpy
// #x e y são iniciados como x = 1.0f e y = 2.0f


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


//Copia valores da CPU para A GPU
// cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
​cudaError_t err;
err = cudaMemcpy ()


}