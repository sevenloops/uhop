// uhop/kernels/cuda/fused_add_norm_act.cu
// Fused kernels for Add + LayerNorm + Activation patterns

extern "C" __global__ void
fused_add_layernorm_gelu_kernel(const float *input, const float *residual,
                                const float *gamma, const float *beta,
                                float *output, int batch_size,
                                int normalized_shape, float eps) {

  int batch_idx = blockIdx.x;

  if (batch_idx >= batch_size)
    return;

  int offset = batch_idx * normalized_shape;

  // Step 1: Add residual
  __shared__ float shared_data[1024];
  for (int i = threadIdx.x; i < normalized_shape; i += blockDim.x) {
    shared_data[i] = input[offset + i] + residual[offset + i];
  }
  __syncthreads();

  // Step 2: Compute mean
  float sum = 0.0f;
  for (int i = 0; i < normalized_shape; ++i) {
    sum += shared_data[i];
  }
  float mean = sum / normalized_shape;

  // Step 3: Compute variance
  float var_sum = 0.0f;
  for (int i = 0; i < normalized_shape; ++i) {
    float diff = shared_data[i] - mean;
    var_sum += diff * diff;
  }
  float variance = var_sum / normalized_shape;
  float inv_std = rsqrtf(variance + eps);

  // Step 4: Normalize, scale, and apply GELU
  for (int i = threadIdx.x; i < normalized_shape; i += blockDim.x) {
    float normalized = (shared_data[i] - mean) * inv_std;
    float scaled = gamma[i] * normalized + beta[i];

    // GELU activation
    float x = scaled;
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    output[offset + i] = 0.5f * x * (1.0f + tanhf(inner));
  }
}

extern "C" __global__ void
fused_add_layernorm_relu_kernel(const float *input, const float *residual,
                                const float *gamma, const float *beta,
                                float *output, int batch_size,
                                int normalized_shape, float eps) {

  int batch_idx = blockIdx.x;

  if (batch_idx >= batch_size)
    return;

  int offset = batch_idx * normalized_shape;

  // Add residual
  __shared__ float shared_data[1024];
  for (int i = threadIdx.x; i < normalized_shape; i += blockDim.x) {
    shared_data[i] = input[offset + i] + residual[offset + i];
  }
  __syncthreads();

  // Compute mean
  float sum = 0.0f;
  for (int i = 0; i < normalized_shape; ++i) {
    sum += shared_data[i];
  }
  float mean = sum / normalized_shape;

  // Compute variance
  float var_sum = 0.0f;
  for (int i = 0; i < normalized_shape; ++i) {
    float diff = shared_data[i] - mean;
    var_sum += diff * diff;
  }
  float variance = var_sum / normalized_shape;
  float inv_std = rsqrtf(variance + eps);

  // Normalize, scale, and apply ReLU
  for (int i = threadIdx.x; i < normalized_shape; i += blockDim.x) {
    float normalized = (shared_data[i] - mean) * inv_std;
    float scaled = gamma[i] * normalized + beta[i];
    output[offset + i] = fmaxf(0.0f, scaled); // ReLU
  }
}

extern "C" __global__ void
fused_add_layernorm_silu_kernel(const float *input, const float *residual,
                                const float *gamma, const float *beta,
                                float *output, int batch_size,
                                int normalized_shape, float eps) {

  int batch_idx = blockIdx.x;

  if (batch_idx >= batch_size)
    return;

  int offset = batch_idx * normalized_shape;

  // Add residual
  __shared__ float shared_data[1024];
  for (int i = threadIdx.x; i < normalized_shape; i += blockDim.x) {
    shared_data[i] = input[offset + i] + residual[offset + i];
  }
  __syncthreads();

  // Compute mean
  float sum = 0.0f;
  for (int i = 0; i < normalized_shape; ++i) {
    sum += shared_data[i];
  }
  float mean = sum / normalized_shape;

  // Compute variance
  float var_sum = 0.0f;
  for (int i = 0; i < normalized_shape; ++i) {
    float diff = shared_data[i] - mean;
    var_sum += diff * diff;
  }
  float variance = var_sum / normalized_shape;
  float inv_std = rsqrtf(variance + eps);

  // Normalize, scale, and apply SiLU
  for (int i = threadIdx.x; i < normalized_shape; i += blockDim.x) {
    float normalized = (shared_data[i] - mean) * inv_std;
    float scaled = gamma[i] * normalized + beta[i];
    // SiLU activation
    output[offset + i] = scaled / (1.0f + expf(-scaled));
  }
}
