// uhop/kernels/cuda/silu.cu
// SiLU (Swish) activation function

extern "C" __global__ void silu_kernel(float *X, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float x = X[idx];
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    X[idx] = x / (1.0f + expf(-x));
  }
}

extern "C" __global__ void swish_kernel(float *X, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float x = X[idx];
    X[idx] = x / (1.0f + expf(-x));
  }
}
