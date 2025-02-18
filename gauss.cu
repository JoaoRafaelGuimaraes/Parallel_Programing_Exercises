

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


// Função para alocar e gerar uma matriz nxn com valores entre 1.0 e 9.0
double **gerarMatriz(int n) {
    double **matriz = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        matriz[i] = (double *)malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            matriz[i][j] = (double)(rand() % 9 + 1); // Valores entre 1.0 e 9.0
        }
    }
    return matriz;
}

// Função para alocar e gerar um vetor de tamanho n com valores aleatórios
double *gerarVetor(int n) {
    double *vetor = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        vetor[i] = (double)(rand() % (10 * n) + 1); // Valores entre 1 e 10n
    }
    return vetor;
}

__global__ void troca_linha(int *idx, double *matriz,int DIM, int coluna, double *col,double * vetor_b){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int index = *idx;

  if (i == index && j<DIM && i>coluna){

      col[j] = matriz[i * DIM + j];
      matriz[i * DIM + j] = matriz[coluna * DIM + j];
      matriz[coluna * DIM + j] = col[j];

     if (j == 0) { // Apenas a primeira thread por bloco deve trocar b
        double temp = vetor_b[index];
        vetor_b[index] = vetor_b[coluna];
        vetor_b[coluna] = temp;
    }

  }


}

__global__ void pivotea(double *matriz,int coluna, double * vetor_b, int DIM){

 int i = blockDim.x * blockIdx.x + threadIdx.x;
 int j = blockDim.y *blockIdx.y + threadIdx.y;

 double value = matriz[coluna*DIM +coluna];

 if(i<DIM && j<DIM && i>coluna){
  double fctr = - matriz[i*DIM +coluna]/ (value);

  matriz[i*DIM + j] +=  fctr* matriz[coluna*DIM +j];

  vetor_b[i] += fctr * vetor_b[coluna];
 }


}


__device__ double atomicMaxFloat(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*) address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__
void max_line(int * idx,double *d_value, double * matriz, int DIM, int coluna){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double old;
    *d_value = -INFINITY;
    if(i < DIM && i>=coluna){
        old = atomicMaxFloat(d_value, fabs(matriz[i*DIM+coluna]));
        if (old < fabs(matriz[i*DIM+coluna]))  // Se encontrou um novo maior valor
            atomicExch(idx, i);  // Atualiza o índice atômico
    }
}

int main(){
    int n = 5;
    double a[5][5] = {
        {7.0, 4.0, 8.0, 5.0, 7.0},
        {3.0, 7.0, 8.0, 5.0, 4.0},
        {8.0, 8.0, 3.0, 6.0, 5.0},
        {2.0, 8.0, 6.0, 2.0, 5.0},
        {1.0, 6.0, 9.0, 1.0, 3.0}
    };

    // Vetor de termos independentes B
    double b[5] = {62.0, 39.0, 37.0, 34.0, 35.0};

    double x[n];

    int a_size = n;
    printf("%d", a_size);
    double *matriz_t, *d_vetor_b, *d_value;
    cudaMalloc(&matriz_t, a_size * a_size * sizeof(double));
    cudaMalloc(&d_vetor_b, a_size * sizeof(double));
    cudaMalloc(&d_value, sizeof(double));

    cudaMemcpy(matriz_t, a, a_size * a_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vetor_b, b, a_size * sizeof(double), cudaMemcpyHostToDevice);

    int *d_index;
    cudaMalloc(&d_index, sizeof(int));
    double * col;
    cudaMalloc(&col, sizeof(double)*a_size);
    cudaDeviceSynchronize();


    dim3 threads_per_block(16, 16);
    int block_qtd = (a_size + 15) / 16;
    dim3 blocks(block_qtd, block_qtd);
    for(int coluna =0; coluna<a_size; coluna++)//A cada linha
    {

        max_line<<<(a_size+255)/256, 256>>>(d_index,d_value, matriz_t,a_size,coluna);//encontra maior valor da linha
        cudaDeviceSynchronize();
        troca_linha<<<blocks, threads_per_block>>>(d_index, matriz_t, a_size,coluna, col,d_vetor_b);
        cudaDeviceSynchronize();
        pivotea<<<blocks, threads_per_block>>>(matriz_t, coluna, d_vetor_b, a_size);
        cudaDeviceSynchronize();

    }




    //cudaMemcpy(&index, d_index, sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(&value, d_value, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(a, matriz_t, a_size * a_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_vetor_b, a_size * sizeof(double), cudaMemcpyDeviceToHost);


    //Resolve o sistema sequencialmente
    int pos = a_size - 1;
    x[pos] = b[pos] / a[pos][pos]; //primeira sol
    double sum;
    for (int i = pos - 1; i >= 0; i--) {
         sum = 0.0;
        for (int j = i + 1; j < a_size; j++) {
            sum += a[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / a[i][i];
    }

    cudaMemcpy(a, matriz_t, a_size * a_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_vetor_b, a_size * sizeof(double), cudaMemcpyDeviceToHost);
    //printf("Index = %d, Value = %lf\n",index,value);

    printf("\nResults\n");
    for (int j = 0; j<a_size;j++){
        printf("x%d = %lf ",j ,x[j]);
    }
    printf("\n");



    //prova de funcionamento

    cudaFree(matriz_t);
    cudaFree(d_index);
    cudaFree(d_vetor_b);
    cudaFree(col);

    return 0;
}
