#include <metal_stdlib>
using namespace metal;

kernel void elementwise_add(device const float* A [[buffer(0)]],
                            device const float* B [[buffer(1)]],
                            device float* Out [[buffer(2)]],
                            constant int& N [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid < N) Out[gid] = A[gid] + B[gid];
}

kernel void elementwise_mul(device const float* A [[buffer(0)]],
                            device const float* B [[buffer(1)]],
                            device float* Out [[buffer(2)]],
                            constant int& N [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid < N) Out[gid] = A[gid] * B[gid];
}

kernel void sigmoid_kernel(device float* X [[buffer(0)]],
                           constant int& N [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid < N) {
        float x = X[gid];
        X[gid] = 1.0f / (1.0f + exp(-x));
    }
}
