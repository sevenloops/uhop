// uhop/kernels/hip/activations.hip
// Activation functions for AMD ROCm

#include <hip/hip_runtime.h>

extern "C" __global__
void relu_kernel(float* X, int N) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < N) {
        float v = X[idx];
        X[idx] = v > 0.0f ? v : 0.0f;
    }
}

extern "C" __global__
void gelu_kernel(float* X, int N) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < N) {
        float x = X[idx];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        X[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

extern "C" __global__
void silu_kernel(float* X, int N) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < N) {
        float x = X[idx];
        X[idx] = x / (1.0f + expf(-x));
    }
}
