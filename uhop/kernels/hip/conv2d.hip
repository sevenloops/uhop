// uhop/kernels/hip/conv2d.hip
// 2D Convolution for AMD ROCm

#include <hip/hip_runtime.h>

extern "C" __global__
void conv2d_kernel(const float* input, const float* weight, const float* bias,
                   float* output, int batch_size, int in_channels, int out_channels,
                   int input_h, int input_w, int kernel_h, int kernel_w,
                   int stride_h, int stride_w, int padding_h, int padding_w) {
    int output_h = (input_h + 2 * padding_h - kernel_h) / stride_h + 1;
    int output_w = (input_w + 2 * padding_w - kernel_w) / stride_w + 1;
    
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int total_output = batch_size * out_channels * output_h * output_w;
    
    if (idx >= total_output) return;
    
    int ow = idx % output_w;
    int oh = (idx / output_w) % output_h;
    int oc = (idx / (output_w * output_h)) % out_channels;
    int b = idx / (output_w * output_h * out_channels);
    
    float sum = bias ? bias[oc] : 0.0f;
    
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = oh * stride_h + kh - padding_h;
                int iw = ow * stride_w + kw - padding_w;
                
                if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                    int in_idx = b * in_channels * input_h * input_w +
                                ic * input_h * input_w + ih * input_w + iw;
                    int w_idx = oc * in_channels * kernel_h * kernel_w +
                               ic * kernel_h * kernel_w + kh * kernel_w + kw;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    
    output[idx] = sum;
}
