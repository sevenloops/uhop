// uhop/kernels/cuda/transpose_dot.cu

extern "C" __global__
void transpose2d(const float* X, float* Y, int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) {
        Y[c * rows + r] = X[r * cols + c];
    }
}

extern "C" __global__
void dot_product(const float* A, const float* B, float* Out, int N) {
    __shared__ float sdata[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.0f;
    if (i < N) v = A[i] * B[i];
    sdata[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(Out, sdata[0]);
}
