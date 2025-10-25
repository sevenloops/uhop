// uhop/kernels/cuda/softmax.cu
// Softmax and LogSoftmax kernels

extern "C" __global__
void softmax_kernel(const float* input, float* output, int batch_size, int num_classes) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int offset = batch_idx * num_classes;
    
    // Find max for numerical stability
    float max_val = input[offset];
    for (int i = 1; i < num_classes; ++i) {
        max_val = fmaxf(max_val, input[offset + i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        float exp_val = expf(input[offset + i] - max_val);
        output[offset + i] = exp_val;
        __syncthreads();
        sum += exp_val;
    }
    
    // Reduce sum across block
    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float total_sum = shared_sum[0];
    
    // Normalize
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        output[offset + i] /= total_sum;
    }
}

extern "C" __global__
void logsoftmax_kernel(const float* input, float* output, int batch_size, int num_classes) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int offset = batch_idx * num_classes;
    
    // Find max for numerical stability
    float max_val = input[offset];
    for (int i = 1; i < num_classes; ++i) {
        max_val = fmaxf(max_val, input[offset + i]);
    }
    
    // Compute log(sum(exp))
    float sum = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
        sum += expf(input[offset + i] - max_val);
    }
    float log_sum_exp = logf(sum) + max_val;
    
    // Compute log probabilities
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        output[offset + i] = input[offset + i] - log_sum_exp;
    }
}
