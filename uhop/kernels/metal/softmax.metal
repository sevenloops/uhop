#include <metal_stdlib>
using namespace metal;

// uhop/kernels/metal/softmax.metal
// Softmax for Metal

kernel void softmax_kernel(device const float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant int& batch_size [[buffer(2)]],
                          constant int& num_classes [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= batch_size) return;
    
    uint offset = gid * num_classes;
    
    // Find max for numerical stability
    float max_val = input[offset];
    for (uint i = 1; i < num_classes; ++i) {
        max_val = max(max_val, input[offset + i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < num_classes; ++i) {
        float exp_val = exp(input[offset + i] - max_val);
        output[offset + i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for (uint i = 0; i < num_classes; ++i) {
        output[offset + i] /= sum;
    }
}

kernel void logsoftmax_kernel(device const float* input [[buffer(0)]],
                              device float* output [[buffer(1)]],
                              constant int& batch_size [[buffer(2)]],
                              constant int& num_classes [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= batch_size) return;
    
    uint offset = gid * num_classes;
    
    // Find max for numerical stability
    float max_val = input[offset];
    for (uint i = 1; i < num_classes; ++i) {
        max_val = max(max_val, input[offset + i]);
    }
    
    // Compute log(sum(exp))
    float sum = 0.0f;
    for (uint i = 0; i < num_classes; ++i) {
        sum += exp(input[offset + i] - max_val);
    }
    float log_sum_exp = log(sum) + max_val;
    
    // Compute log probabilities
    for (uint i = 0; i < num_classes; ++i) {
        output[offset + i] = input[offset + i] - log_sum_exp;
    }
}
