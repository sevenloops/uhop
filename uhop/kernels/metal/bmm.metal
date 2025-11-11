#include <metal_stdlib>
using namespace metal;

// uhop/kernels/metal/bmm.metal
// Batched Matrix Multiplication for Metal

kernel void bmm_kernel(device const float* A [[buffer(0)]],
                       device const float* B [[buffer(1)]],
                       device float* C [[buffer(2)]],
                       constant int& batch_size [[buffer(3)]],
                       constant int& M [[buffer(4)]],
                       constant int& N [[buffer(5)]],
                       constant int& K [[buffer(6)]],
                       uint3 gid [[thread_position_in_grid]]) {
    uint b = gid.z;
    uint row = gid.y;
    uint col = gid.x;

    if (b >= batch_size || row >= M || col >= K) return;

    uint a_offset = b * M * N;
    uint b_offset = b * N * K;
    uint c_offset = b * M * K;

    float sum = 0.0f;
    for (uint i = 0; i < N; ++i) {
        sum += A[a_offset + row * N + i] * B[b_offset + i * K + col];
    }
    C[c_offset + row * K + col] = sum;
}
