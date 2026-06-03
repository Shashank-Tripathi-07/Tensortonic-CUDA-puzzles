#include <cuda_runtime.h>

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int M, int N) {
    int  i = threadIdx.x + blockIdx.x*blockDim.x; 
    int  j = threadIdx.y + blockIdx.y*blockDim.y; 

    if (i < N && j < M){
        int z = j*N + i; 
        C[z] = A[z] + B[z];  
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matrix_add_kernel<<<blocks, threads>>>(A, B, C, M, N);
    cudaDeviceSynchronize();
}

// Matrix Addition but with a twist :) 
// I hope you like it ! 
