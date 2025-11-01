// uhop/kernels/opencl/activations.cl
// Activation functions for OpenCL

__kernel void relu_kernel(__global float *X, int N) {
  int idx = get_global_id(0);
  if (idx < N) {
    float v = X[idx];
    X[idx] = (v > 0.0f) ? v : 0.0f;
  }
}

__kernel void gelu_kernel(__global float *X, int N) {
  int idx = get_global_id(0);
  if (idx < N) {
    float x = X[idx];
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    X[idx] = 0.5f * x * (1.0f + tanh(inner));
  }
}

__kernel void silu_kernel(__global float *X, int N) {
  int idx = get_global_id(0);
  if (idx < N) {
    float x = X[idx];
    X[idx] = x / (1.0f + exp(-x));
  }
}
