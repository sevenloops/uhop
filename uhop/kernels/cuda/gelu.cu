// uhop/kernels/cuda/gelu.cu
// GELU activation function

extern "C" __global__ void gelu_kernel(float *X, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float x = X[idx];
    // GELU(x) = x * Phi(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3); // sqrt(2/pi) ≈ 0.797885
    X[idx] = 0.5f * x * (1.0f + tanhf(inner));
  }
}

extern "C" __global__ void gelu_exact_kernel(float *X, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float x = X[idx];
    // Exact GELU using erf
    X[idx] = 0.5f * x * (1.0f + erff(x * 0.7071067812f)); // 1/sqrt(2)
  }
}
