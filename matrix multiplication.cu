#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int column = threadIdx.x + blockIdx.x*blockDim.x; 
    int row = threadIdx.y + blockIdx.y*blockDim.y; 

    if(row < M && column < N){ 
        float sum = 0.0f; 

        for (int k=0; k < K ; k++){
            sum += A[row*K+k]*B[k*N+column]; 
        }

        C[row*N + column] = sum; 
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

/* Hey, so I'm gonna explain how this code works. 

You remember we do, M*N and N*M multiplication only when the N is same in both matrices. 
Otherwise we don't do it. (idk who tf decided it but I trust my math prof and humanity with this one !, it can be a good evening research)

so, we use this here to do matrix multiplication. 

We run one thread and each thread is responsible for calculating one element of the output matrix.
So, we calculate the row and column of the output matrix that this thread is responsible for (by running it in a loop)

Then we check if the row and column are within the bounds of the output matrix (by using the if statement).

( Mind this that if we don't check it the GPU just runs out of the bounds and goes for all the threads in the SM 
and calculates the answers in that and fills the final output with wrong answer. )


If they are, we calculate the dot product of the row of the first matrix and the column
of the second matrix to get the value of the output matrix at that position.

We do this by iterating over the elements of the row and column and multiplying them together and summing them up. 
(Looks easy but it's a bit tricky to understand at first, but once you get it, it's pretty straightforward.)
(also it looks like magic but it isn't, it's just a lot of threads doing a lot of calculations in parallel, which is what GPUs are really good at.)

Finally, we store the result in the output matrix.

HAIL CUDA !! and low-level programming. 
*/  
