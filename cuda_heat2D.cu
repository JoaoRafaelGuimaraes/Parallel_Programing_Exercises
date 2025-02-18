#include <stdio.h>

// Objetivo: Implementar um algoritmo para simular transferencia de calor utilizando CUDA 

//Aproveitar memória local dos SM a partir do uso de variáveis shared?
__global__
void inicia_matriz(int dim_x, int dim_y, float *heat_m){

    float radius2 = (dim_x/6) * (dim_x/6);
    float distance2;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i<dim_x && j<dim_y){
    
        //Distance² to the center point (dim_x/2, dim_y/2) 
        
        distance2 = (i - dim_x/2)*(i - dim_x/2) + (j-dim_y/2)*(j-dim_y/2);


        if (distance2<radius2)
            heat_m[i*dim_x + j] = 62.0;
        else    
            heat_m[i*dim_x + j] = 5.0;
    }

}

__global__
void heat_propagation(int dim_x, int dim_y,float *heat_m, float *heat_m_temp, float dx2, float dy2, float a, float dt){

    // Unp1[i,j] = Un[i,j] + dt * a * ((Un[i-1,j] - 2*Un[i,j] + Un[i+1,j]) / dx2 + (Un[i,j-1] - 2*Un[i,j] + Un[i,j+1]) / dy2)

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    
    if (i>0 && i<dim_x-1 && j<dim_y-1 && j>0){

        float Unij = heat_m[i*dim_x + j];
        float Unim1j = heat_m[(i-1)*dim_x + j];
        float Unip1j = heat_m[(i+1)*dim_x + j];
        float Unijp1 = heat_m[i*dim_x + j+1];
        float Unijm1 = heat_m[i*dim_x + j-1];


        heat_m_temp[j * dim_x + i] = Unij + dt * a * 
                            ( (Unim1j - 2.0*Unij + Unip1j)/dx2 + (Unijm1 - 2.0*Unij + Unijp1)/dy2 );


    }
   
}

int main(){

    //Algoritmo - Dividir matriz original em matrizes menores, compostas por blocos de threads de tamanho 16x16
    const float a = 0.5;     // Diffusion constant

	const float dx2 = 0.01*0.01;   // Horizontal grid spacing
	const float dy2 = 0.01*0.01;   // Vertical grid spacing

    const float dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2)); // Largest stable time step
	const int numSteps = 5000;                             // Number of time steps
    //Dimensões da minha matriz
    int dim_x = 200;
    int dim_y = 200;
    

    //Iniciar a matriz (Por que não fazer isso em kernel?)

    float *heat_m, *heat_m_temp, *temp;

    int size = dim_x*dim_y;
    // heat_m = (float *)calloc(dim_x*dim_y,size*sizeof(float) );
    cudaMalloc(&heat_m, size*sizeof(float));

    dim3 threads_per_block(16,16);
    dim3 blocks((dim_x+15)/16, (dim_y+15)/16);
    
    inicia_matriz<<<blocks,threads_per_block>>>(dim_x, dim_y, heat_m);
    

    
    //Prosseguir para cálculo constante de propagação de calor
    cudaMalloc(&heat_m_temp, size*sizeof(float));

    for (int i = 0; i<numSteps; i++){


        heat_propagation<<<blocks, threads_per_block>>>(dim_x,dim_y,heat_m,heat_m_temp,dx2,dy2,a,dt);
        
        
        cudaDeviceSynchronize();
        temp = heat_m; heat_m = heat_m_temp; heat_m_temp = temp;
    }


    cudaFree(heat_m);
    cudaFree(heat_m_temp);

}