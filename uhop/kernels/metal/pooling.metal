#include <metal_stdlib>
using namespace metal;

// uhop/kernels/metal/pooling.metal
// Pooling operations for Metal

kernel void maxpool2d_kernel(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant int& batch_size [[buffer(2)]],
                             constant int& channels [[buffer(3)]],
                             constant int& input_h [[buffer(4)]],
                             constant int& input_w [[buffer(5)]],
                             constant int& kernel_h [[buffer(6)]],
                             constant int& kernel_w [[buffer(7)]],
                             constant int& stride_h [[buffer(8)]],
                             constant int& stride_w [[buffer(9)]],
                             constant int& padding_h [[buffer(10)]],
                             constant int& padding_w [[buffer(11)]],
                             uint gid [[thread_position_in_grid]]) {
    int output_h = (input_h + 2 * padding_h - kernel_h) / stride_h + 1;
    int output_w = (input_w + 2 * padding_w - kernel_w) / stride_w + 1;

    int total_output = batch_size * channels * output_h * output_w;

    if (gid >= total_output) return;

    int ow = gid % output_w;
    int oh = (gid / output_w) % output_h;
    int c = (gid / (output_w * output_h)) % channels;
    int b = gid / (output_w * output_h * channels);

    float max_val = -1e38f;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int ih = oh * stride_h + kh - padding_h;
            int iw = ow * stride_w + kw - padding_w;

            if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                int in_idx = b * channels * input_h * input_w +
                            c * input_h * input_w +
                            ih * input_w + iw;
                max_val = max(max_val, input[in_idx]);
            }
        }
    }

    output[gid] = max_val;
}

kernel void avgpool2d_kernel(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant int& batch_size [[buffer(2)]],
                             constant int& channels [[buffer(3)]],
                             constant int& input_h [[buffer(4)]],
                             constant int& input_w [[buffer(5)]],
                             constant int& kernel_h [[buffer(6)]],
                             constant int& kernel_w [[buffer(7)]],
                             constant int& stride_h [[buffer(8)]],
                             constant int& stride_w [[buffer(9)]],
                             constant int& padding_h [[buffer(10)]],
                             constant int& padding_w [[buffer(11)]],
                             constant bool& count_include_pad [[buffer(12)]],
                             uint gid [[thread_position_in_grid]]) {
    int output_h = (input_h + 2 * padding_h - kernel_h) / stride_h + 1;
    int output_w = (input_w + 2 * padding_w - kernel_w) / stride_w + 1;

    int total_output = batch_size * channels * output_h * output_w;

    if (gid >= total_output) return;

    int ow = gid % output_w;
    int oh = (gid / output_w) % output_h;
    int c = (gid / (output_w * output_h)) % channels;
    int b = gid / (output_w * output_h * channels);

    float sum = 0.0f;
    int count = 0;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int ih = oh * stride_h + kh - padding_h;
            int iw = ow * stride_w + kw - padding_w;

            if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                int in_idx = b * channels * input_h * input_w +
                            c * input_h * input_w +
                            ih * input_w + iw;
                sum += input[in_idx];
                count++;
            } else if (count_include_pad) {
                count++;
            }
        }
    }

    output[gid] = sum / float((count > 0) ? count : 1);
}
