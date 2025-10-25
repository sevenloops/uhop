// uhop/kernels/hip/reduce.hip
#include <hip/hip_runtime.h>

extern "C" __global__
void reduce_sum_atomic(const float* X, float* Out, int N) {
    __shared__ float block_sum;
    if (hipThreadIdx_x == 0) block_sum = 0.0f;
    __syncthreads();
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    float v = (i < N) ? X[i] : 0.0f;
    atomicAdd(&block_sum, v);
    __syncthreads();
    if (hipThreadIdx_x == 0) atomicAdd(Out, block_sum);
}
