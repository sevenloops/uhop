// uhop/kernels/cuda/einsum.cu
// Generic einsum kernels for common patterns

// Matrix multiplication: "ij,jk->ik"
extern "C" __global__ void einsum_matmul_kernel(const float *A, const float *B,
                                                float *C, int I, int J, int K) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < I && k < K) {
    float sum = 0.0f;
    for (int j = 0; j < J; ++j) {
      sum += A[i * J + j] * B[j * K + k];
    }
    C[i * K + k] = sum;
  }
}

// Batch matrix multiplication: "bij,bjk->bik"
extern "C" __global__ void einsum_bmm_kernel(const float *A, const float *B,
                                             float *C, int B_size, int I, int J,
                                             int K) {
  int b = blockIdx.z;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.x * blockDim.x + threadIdx.x;

  if (b < B_size && i < I && k < K) {
    int a_offset = b * I * J;
    int b_offset = b * J * K;
    int c_offset = b * I * K;

    float sum = 0.0f;
    for (int j = 0; j < J; ++j) {
      sum += A[a_offset + i * J + j] * B[b_offset + j * K + k];
    }
    C[c_offset + i * K + k] = sum;
  }
}

// Tensor contraction: "ijk,ikl->ijl"
extern "C" __global__ void einsum_tensor_contract_kernel(const float *A,
                                                         const float *B,
                                                         float *C, int I, int J,
                                                         int K, int L) {
  int i = blockIdx.z;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int l = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < I && j < J && l < L) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += A[i * J * K + j * K + k] * B[i * K * L + k * L + l];
    }
    C[i * J * L + j * L + l] = sum;
  }
}
