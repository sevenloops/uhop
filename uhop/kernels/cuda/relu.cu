// uhop/kernels/cuda/relu.cu
extern "C" __global__
void relu_kernel(float* X, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float v = X[idx];
        X[idx] = v > 0.0f ? v : 0.0f;
    }
}
