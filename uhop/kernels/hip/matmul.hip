// uhop/kernels/hip/matmul.hip
// Matrix Multiplication for AMD ROCm

#include <hip/hip_runtime.h>

extern "C" __global__
void matmul_kernel(const float* A, const float* B, float* C, int N, int M, int K) {
    int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (row < N && col < K) {
        float s = 0.0f;
        for (int i = 0; i < M; ++i) {
            s += A[row * M + i] * B[i * K + col];
        }
        C[row * K + col] = s;
    }
}
