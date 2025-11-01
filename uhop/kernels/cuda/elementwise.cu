// uhop/kernels/cuda/elementwise.cu
// Common elementwise operations for CUDA

extern "C" __global__ void elementwise_add(const float *A, const float *B,
                                           float *Out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    Out[i] = A[i] + B[i];
}

extern "C" __global__ void elementwise_sub(const float *A, const float *B,
                                           float *Out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    Out[i] = A[i] - B[i];
}

extern "C" __global__ void elementwise_mul(const float *A, const float *B,
                                           float *Out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    Out[i] = A[i] * B[i];
}

extern "C" __global__ void elementwise_div(const float *A, const float *B,
                                           float *Out, int N, float eps) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float d = B[i];
    if (fabsf(d) < eps)
      d = (d >= 0.0f ? eps : -eps);
    Out[i] = A[i] / d;
  }
}

extern "C" __global__ void elementwise_pow(const float *A, const float *B,
                                           float *Out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    Out[i] = powf(A[i], B[i]);
}

extern "C" __global__ void elementwise_max(const float *A, const float *B,
                                           float *Out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    Out[i] = fmaxf(A[i], B[i]);
}

extern "C" __global__ void elementwise_min(const float *A, const float *B,
                                           float *Out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    Out[i] = fminf(A[i], B[i]);
}

extern "C" __global__ void elementwise_leakyrelu(const float *X, float *Out,
                                                 int N, float alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float v = X[i];
    Out[i] = v > 0.0f ? v : alpha * v;
  }
}

extern "C" __global__ void sigmoid_kernel(float *X, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float x = X[i];
    X[i] = 1.0f / (1.0f + expf(-x));
  }
}

extern "C" __global__ void tanh_kernel(float *X, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float x = X[i];
    float epos = expf(x);
    float eneg = expf(-x);
    X[i] = (epos - eneg) / (epos + eneg);
  }
}

extern "C" __global__ void elementwise_exp(const float *X, float *Out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    Out[i] = expf(X[i]);
}

extern "C" __global__ void elementwise_log(const float *X, float *Out, int N,
                                           float eps) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    Out[i] = logf(fmaxf(fabsf(X[i]), eps));
}

extern "C" __global__ void elementwise_sqrt(const float *X, float *Out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    Out[i] = sqrtf(fabsf(X[i]));
}
