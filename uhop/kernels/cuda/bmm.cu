// uhop/kernels/cuda/bmm.cu
// Batched Matrix Multiplication kernel

extern "C" __global__
void bmm_kernel(const float* A, const float* B, float* C, 
                int batch_size, int M, int N, int K) {
    // Batch index
    int b = blockIdx.z;
    if (b >= batch_size) return;
    
    // Matrix indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        int a_offset = b * M * N;
        int b_offset = b * N * K;
        int c_offset = b * M * K;
        
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[a_offset + row * N + i] * B[b_offset + i * K + col];
        }
        C[c_offset + row * K + col] = sum;
    }
}
