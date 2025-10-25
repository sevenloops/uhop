// uhop/kernels/cuda/layernorm.cu
// Layer Normalization kernel

extern "C" __global__
void layernorm_kernel(const float* input, const float* gamma, const float* beta,
                      float* output, int batch_size, int normalized_shape, float eps) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int offset = batch_idx * normalized_shape;
    
    // Compute mean
    float sum = 0.0f;
    for (int i = 0; i < normalized_shape; ++i) {
        sum += input[offset + i];
    }
    float mean = sum / normalized_shape;
    
    // Compute variance
    float var_sum = 0.0f;
    for (int i = 0; i < normalized_shape; ++i) {
        float diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / normalized_shape;
    float inv_std = rsqrtf(variance + eps);
    
    // Normalize and scale
    for (int i = threadIdx.x; i < normalized_shape; i += blockDim.x) {
        float normalized = (input[offset + i] - mean) * inv_std;
        output[offset + i] = gamma[i] * normalized + beta[i];
    }
}
