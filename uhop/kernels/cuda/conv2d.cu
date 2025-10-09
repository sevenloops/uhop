// uhop/kernels/cuda/conv2d.cu
// Very small naive conv2d for demonstration
extern "C" __global__
void conv2d_kernel(const float* input, const float* kernel, float* output,
                   int H, int W, int KH, int KW, int outH, int outW) {
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    if (oy < outH && ox < outW) {
        float s = 0.0f;
        for (int ky = 0; ky < KH; ++ky) {
            for (int kx = 0; kx < KW; ++kx) {
                int iy = oy + ky;
                int ix = ox + kx;
                s += input[iy * W + ix] * kernel[ky * KW + kx];
            }
        }
        output[oy * outW + ox] = s;
    }
}
