// uhop/kernels/hip/layernorm.hip
// Layer Normalization for AMD ROCm

#include <hip/hip_runtime.h>

extern "C" __global__
void layernorm_kernel(const float* input, const float* gamma, const float* beta,
                      float* output, int batch_size, int normalized_shape, float eps) {
    int batch_idx = hipBlockIdx_x;
    
    if (batch_idx >= batch_size) return;
    
    int offset = batch_idx * normalized_shape;
    
    float sum = 0.0f;
    for (int i = 0; i < normalized_shape; ++i) {
        sum += input[offset + i];
    }
    float mean = sum / normalized_shape;
    
    float var_sum = 0.0f;
    for (int i = 0; i < normalized_shape; ++i) {
        float diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / normalized_shape;
    float inv_std = rsqrtf(variance + eps);
    
    for (int i = hipThreadIdx_x; i < normalized_shape; i += hipBlockDim_x) {
        float normalized = (input[offset + i] - mean) * inv_std;
        output[offset + i] = gamma[i] * normalized + beta[i];
    }
}
