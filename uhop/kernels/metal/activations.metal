#include <metal_stdlib>
using namespace metal;

// uhop/kernels/metal/activations.metal
// Activation functions for Metal

kernel void relu_kernel(device float* X [[buffer(0)]],
                        constant int& N [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < N) {
        float v = X[gid];
        X[gid] = max(v, 0.0f);
    }
}

kernel void gelu_kernel(device float* X [[buffer(0)]],
                        constant int& N [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < N) {
        float x = X[gid];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        X[gid] = 0.5f * x * (1.0f + tanh(inner));
    }
}

kernel void silu_kernel(device float* X [[buffer(0)]],
                        constant int& N [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < N) {
        float x = X[gid];
        X[gid] = x / (1.0f + exp(-x));
    }
}
