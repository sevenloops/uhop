// uhop/kernels/cuda/batchnorm.cu
// Batch Normalization kernel

extern "C" __global__
void batchnorm_kernel(const float* input, const float* gamma, const float* beta,
                      const float* running_mean, const float* running_var,
                      float* output, int batch_size, int channels, int spatial_size,
                      float eps, bool training) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * spatial_size;
    
    if (idx >= total_size) return;
    
    int c = (idx / spatial_size) % channels;
    
    float mean, var;
    if (training) {
        // Would need reduction kernels for proper training mode
        // For now, use running stats
        mean = running_mean[c];
        var = running_var[c];
    } else {
        mean = running_mean[c];
        var = running_var[c];
    }
    
    float inv_std = rsqrtf(var + eps);
    float normalized = (input[idx] - mean) * inv_std;
    output[idx] = gamma[c] * normalized + beta[c];
}

extern "C" __global__
void batchnorm_2d_kernel(const float* input, const float* gamma, const float* beta,
                         const float* running_mean, const float* running_var,
                         float* output, int batch_size, int channels, int height, int width,
                         float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = height * width;
    int total_size = batch_size * channels * spatial_size;
    
    if (idx >= total_size) return;
    
    int c = (idx / spatial_size) % channels;
    
    float mean = running_mean[c];
    float var = running_var[c];
    float inv_std = rsqrtf(var + eps);
    
    float normalized = (input[idx] - mean) * inv_std;
    output[idx] = gamma[c] * normalized + beta[c];
}
