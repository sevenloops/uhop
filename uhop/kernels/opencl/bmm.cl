// uhop/kernels/opencl/bmm.cl
// Batched Matrix Multiplication for OpenCL

__kernel void bmm_kernel(__global const float* A,
                         __global const float* B,
                         __global float* C,
                         int batch_size, int M, int N, int K) {
    int b = get_global_id(2);
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    if (b >= batch_size || row >= M || col >= K) return;
    
    int a_offset = b * M * N;
    int b_offset = b * N * K;
    int c_offset = b * M * K;
    
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += A[a_offset + row * N + i] * B[b_offset + i * K + col];
    }
    C[c_offset + row * K + col] = sum;
}
