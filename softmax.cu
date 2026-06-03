#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {

    __shared__ float sdata[256];

    int tid = threadIdx.x;

    // Find max

    float local_max = -FLT_MAX;

    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    float max_val = sdata[0];
    __syncthreads();

    // Find sum

    float local_sum = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(input[i] - max_val);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float sum_val = sdata[0];
    __syncthreads();

    // Write output

    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - max_val) / sum_val;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;

    // Softmax needs one block to cooperate on the whole array
    int blocks = 1;

    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}

// revising this for the next time, absolutely cool question ! 
// I spent 3 hours trying to figure out a solution and ended up discussing it with another engineer. 
// Tell me your experience human ! 
// :) 