// uhop/kernels/opencl/softmax.cl
// Softmax for OpenCL

__kernel void softmax_kernel(__global const float *input,
                             __global float *output, int batch_size,
                             int num_classes) {
  int batch_idx = get_global_id(0);

  if (batch_idx >= batch_size)
    return;

  int offset = batch_idx * num_classes;

  // Find max for numerical stability
  float max_val = input[offset];
  for (int i = 1; i < num_classes; ++i) {
    max_val = fmax(max_val, input[offset + i]);
  }

  // Compute exp and sum
  float sum = 0.0f;
  for (int i = 0; i < num_classes; ++i) {
    float exp_val = exp(input[offset + i] - max_val);
    output[offset + i] = exp_val;
    sum += exp_val;
  }

  // Normalize
  for (int i = 0; i < num_classes; ++i) {
    output[offset + i] /= sum;
  }
}

__kernel void logsoftmax_kernel(__global const float *input,
                                __global float *output, int batch_size,
                                int num_classes) {
  int batch_idx = get_global_id(0);

  if (batch_idx >= batch_size)
    return;

  int offset = batch_idx * num_classes;

  // Find max for numerical stability
  float max_val = input[offset];
  for (int i = 1; i < num_classes; ++i) {
    max_val = fmax(max_val, input[offset + i]);
  }

  // Compute log(sum(exp))
  float sum = 0.0f;
  for (int i = 0; i < num_classes; ++i) {
    sum += exp(input[offset + i] - max_val);
  }
  float log_sum_exp = log(sum) + max_val;

  // Compute log probabilities
  for (int i = 0; i < num_classes; ++i) {
    output[offset + i] = input[offset + i] - log_sum_exp;
  }
}
