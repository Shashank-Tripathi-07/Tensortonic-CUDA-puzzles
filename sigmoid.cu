#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* input, float* output, int N) {
    int i = threadIdx.x + blockDim.x*blockIdx.x ; 

    float j = (input[i]); // we need this to make the code simpler. 

    if (i < N){
        output[i] = (expf(j) - expf (-j)) / (expf (j) + expf (-j)); 
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    tanh_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}