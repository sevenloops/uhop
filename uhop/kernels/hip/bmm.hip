// uhop/kernels/hip/bmm.hip
// Batched Matrix Multiplication kernel for AMD ROCm

#include <hip/hip_runtime.h>

extern "C" __global__
void bmm_kernel(const float* A, const float* B, float* C,
                int batch_size, int M, int N, int K) {
    int b = hipBlockIdx_z;
    if (b >= batch_size) return;

    int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (row < M && col < K) {
        int a_offset = b * M * N;
        int b_offset = b * N * K;
        int c_offset = b * M * K;

        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[a_offset + row * N + i] * B[b_offset + i * K + col];
        }
        C[c_offset + row * K + col] = sum;
    }
}
