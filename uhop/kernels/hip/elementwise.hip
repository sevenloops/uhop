// uhop/kernels/hip/elementwise.hip
#include <hip/hip_runtime.h>

extern "C" __global__
void elementwise_add(const float* A, const float* B, float* Out, int N) {
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i < N) Out[i] = A[i] + B[i];
}

extern "C" __global__
void elementwise_sub(const float* A, const float* B, float* Out, int N) {
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i < N) Out[i] = A[i] - B[i];
}

extern "C" __global__
void elementwise_mul(const float* A, const float* B, float* Out, int N) {
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i < N) Out[i] = A[i] * B[i];
}

extern "C" __global__
void sigmoid_kernel(float* X, int N) {
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i < N) {
        float x = X[i];
        X[i] = 1.0f / (1.0f + expf(-x));
    }
}
