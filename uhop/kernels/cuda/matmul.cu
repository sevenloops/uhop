// uhop/kernels/cuda/matmul.cu
extern "C" __global__
void matmul_kernel(const float* A, const float* B, float* C, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < K) {
        float s = 0.0f;
        for (int i = 0; i < M; ++i) {
            s += A[row * M + i] * B[i * K + col];
        }
        C[row * K + col] = s;
    }
}
