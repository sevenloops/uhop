#include <metal_stdlib>
using namespace metal;

// uhop/kernels/metal/layernorm.metal
// Layer Normalization for Metal

kernel void layernorm_kernel(device const float* input [[buffer(0)]],
                             device const float* gamma [[buffer(1)]],
                             device const float* beta [[buffer(2)]],
                             device float* output [[buffer(3)]],
                             constant int& batch_size [[buffer(4)]],
                             constant int& normalized_shape [[buffer(5)]],
                             constant float& eps [[buffer(6)]],
                             uint gid [[thread_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             threadgroup float* shared_data [[threadgroup(0)]]) {
    uint batch_idx = gid / normalized_shape;
    
    if (batch_idx >= batch_size) return;
    
    uint offset = batch_idx * normalized_shape;
    
    // Compute mean
    float sum = 0.0f;
    for (uint i = 0; i < normalized_shape; ++i) {
        sum += input[offset + i];
    }
    float mean = sum / float(normalized_shape);
    
    // Compute variance
    float var_sum = 0.0f;
    for (uint i = 0; i < normalized_shape; ++i) {
        float diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / float(normalized_shape);
    float inv_std = rsqrt(variance + eps);
    
    // Normalize and scale
    uint i = tid;
    while (i < normalized_shape) {
        float normalized = (input[offset + i] - mean) * inv_std;
        output[offset + i] = gamma[i] * normalized + beta[i];
        i += 256;  // threadgroup size
    }
}
