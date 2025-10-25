// uhop/kernels/opencl/layernorm.cl
// Layer Normalization for OpenCL

__kernel void layernorm_kernel(__global const float* input,
                               __global const float* gamma,
                               __global const float* beta,
                               __global float* output,
                               int batch_size, int normalized_shape, float eps) {
    int batch_idx = get_global_id(0);
    
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
    float inv_std = rsqrt(variance + eps);
    
    // Normalize and scale
    for (int i = 0; i < normalized_shape; ++i) {
        float normalized = (input[offset + i] - mean) * inv_std;
        output[offset + i] = gamma[i] * normalized + beta[i];
    }
}
