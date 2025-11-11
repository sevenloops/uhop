#include <metal_stdlib>
using namespace metal;

// uhop/kernels/metal/matmul.metal
// Matrix Multiplication for Metal

kernel void matmul_kernel(device const float* A [[buffer(0)]],
                          device const float* B [[buffer(1)]],
                          device float* C [[buffer(2)]],
                          constant int& N [[buffer(3)]],
                          constant int& M [[buffer(4)]],
                          constant int& K [[buffer(5)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;

    if (row < N && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += A[row * M + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}
