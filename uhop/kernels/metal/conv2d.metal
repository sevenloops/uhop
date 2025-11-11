#include <metal_stdlib>
using namespace metal;

// uhop/kernels/metal/conv2d.metal
// 2D Convolution for Metal

kernel void conv2d_kernel(device const float* input [[buffer(0)]],
                          device const float* weight [[buffer(1)]],
                          device const float* bias [[buffer(2)]],
                          device float* output [[buffer(3)]],
                          constant int& batch_size [[buffer(4)]],
                          constant int& in_channels [[buffer(5)]],
                          constant int& out_channels [[buffer(6)]],
                          constant int& input_h [[buffer(7)]],
                          constant int& input_w [[buffer(8)]],
                          constant int& kernel_h [[buffer(9)]],
                          constant int& kernel_w [[buffer(10)]],
                          constant int& stride_h [[buffer(11)]],
                          constant int& stride_w [[buffer(12)]],
                          constant int& padding_h [[buffer(13)]],
                          constant int& padding_w [[buffer(14)]],
                          uint gid [[thread_position_in_grid]]) {
    int output_h = (input_h + 2 * padding_h - kernel_h) / stride_h + 1;
    int output_w = (input_w + 2 * padding_w - kernel_w) / stride_w + 1;

    int total_output = batch_size * out_channels * output_h * output_w;

    if (gid >= total_output) return;

    int ow = gid % output_w;
    int oh = (gid / output_w) % output_h;
    int oc = (gid / (output_w * output_h)) % out_channels;
    int b = gid / (output_w * output_h * out_channels);

    float sum = bias[oc];

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

    output[gid] = sum;
}
