// uhop/kernels/cuda/attention.cu
// Scaled Dot-Product Attention kernel

extern "C" __global__
void scaled_dot_product_attention_kernel(
    const float* Q, const float* K, const float* V,
    float* output, float* scores,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal) {
    
    int b = blockIdx.z / num_heads;
    int h = blockIdx.z % num_heads;
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // query position
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // key position
    
    if (b >= batch_size || i >= seq_len || j >= seq_len) return;
    
    int offset = (b * num_heads + h) * seq_len * head_dim;
    
    // Compute Q @ K^T
    if (i < seq_len && j < seq_len) {
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            sum += Q[offset + i * head_dim + d] * K[offset + j * head_dim + d];
        }
        sum *= scale;
        
        // Apply causal mask if needed
        if (causal && j > i) {
            sum = -1e10f;  // -inf
        }
        
        int score_offset = (b * num_heads + h) * seq_len * seq_len;
        scores[score_offset + i * seq_len + j] = sum;
    }
}

extern "C" __global__
void attention_softmax_kernel(float* scores, int batch_size, int num_heads, int seq_len) {
    int b = blockIdx.z / num_heads;
    int h = blockIdx.z % num_heads;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= batch_size || i >= seq_len) return;
    
    int offset = (b * num_heads + h) * seq_len * seq_len + i * seq_len;
    
    // Find max for numerical stability
    float max_val = -1e10f;
    for (int j = 0; j < seq_len; ++j) {
        max_val = fmaxf(max_val, scores[offset + j]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        scores[offset + j] = expf(scores[offset + j] - max_val);
        sum += scores[offset + j];
    }
    
    // Normalize
    for (int j = 0; j < seq_len; ++j) {
        scores[offset + j] /= sum;
    }
}

extern "C" __global__
void attention_output_kernel(
    const float* scores, const float* V, float* output,
    int batch_size, int num_heads, int seq_len, int head_dim) {
    
    int b = blockIdx.z / num_heads;
    int h = blockIdx.z % num_heads;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= batch_size || i >= seq_len || d >= head_dim) return;
    
    int score_offset = (b * num_heads + h) * seq_len * seq_len + i * seq_len;
    int v_offset = (b * num_heads + h) * seq_len * head_dim;
    int out_offset = (b * num_heads + h) * seq_len * head_dim;
    
    float sum = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        sum += scores[score_offset + j] * V[v_offset + j * head_dim + d];
    }
    
    output[out_offset + i * head_dim + d] = sum;
}
