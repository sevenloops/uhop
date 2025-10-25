// uhop/kernels/metal/conv2d_backward.metal
// Compute input and weight gradients for 2D convolution (NCHW) for Metal
#include <metal_stdlib>
using namespace metal;

kernel void conv2d_input_grad(
    device const float* grad_out [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    device float* grad_in        [[buffer(2)]],
    constant int& N              [[buffer(3)]],
    constant int& Cin            [[buffer(4)]],
    constant int& H              [[buffer(5)]],
    constant int& W              [[buffer(6)]],
    constant int& Cout           [[buffer(7)]],
    constant int& KH             [[buffer(8)]],
    constant int& KW             [[buffer(9)]],
    constant int& outH           [[buffer(10)]],
    constant int& outW           [[buffer(11)]],
    constant int& stride_h       [[buffer(12)]],
    constant int& stride_w       [[buffer(13)]],
    constant int& pad_h          [[buffer(14)]],
    constant int& pad_w          [[buffer(15)]],
    uint gid [[thread_position_in_grid]]) {

    int idx = int(gid);
    int total = N * Cin * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % Cin;
    int n = idx / (W * H * Cin);

    float sum = 0.0f;
    for (int co = 0; co < Cout; ++co) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                int oh_nom = h + pad_h - kh;
                int ow_nom = w + pad_w - kw;
                if ((oh_nom % stride_h) == 0 && (ow_nom % stride_w) == 0) {
                    int oh = oh_nom / stride_h;
                    int ow = ow_nom / stride_w;
                    if (oh >= 0 && oh < outH && ow >= 0 && ow < outW) {
                        int go_idx = ((n * Cout + co) * outH + oh) * outW + ow;
                        int w_idx = ((co * Cin + c) * KH + kh) * KW + kw;
                        sum += grad_out[go_idx] * weight[w_idx];
                    }
                }
            }
        }
    }
    grad_in[idx] = sum;
}

kernel void conv2d_weight_grad(
    device const float* input    [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_w         [[buffer(2)]],
    constant int& N              [[buffer(3)]],
    constant int& Cin            [[buffer(4)]],
    constant int& H              [[buffer(5)]],
    constant int& W              [[buffer(6)]],
    constant int& Cout           [[buffer(7)]],
    constant int& KH             [[buffer(8)]],
    constant int& KW             [[buffer(9)]],
    constant int& outH           [[buffer(10)]],
    constant int& outW           [[buffer(11)]],
    constant int& stride_h       [[buffer(12)]],
    constant int& stride_w       [[buffer(13)]],
    constant int& pad_h          [[buffer(14)]],
    constant int& pad_w          [[buffer(15)]],
    uint gid [[thread_position_in_grid]]) {

    int idx = int(gid);
    int total = Cout * Cin * KH * KW;
    if (idx >= total) return;

    int kw = idx % KW;
    int kh = (idx / KW) % KH;
    int c = (idx / (KW * KH)) % Cin;
    int co = idx / (KW * KH * Cin);

    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < outH; ++oh) {
            for (int ow = 0; ow < outW; ++ow) {
                int ih = oh * stride_h + kh - pad_h;
                int iw = ow * stride_w + kw - pad_w;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    int in_idx = ((n * Cin + c) * H + ih) * W + iw;
                    int go_idx = ((n * Cout + co) * outH + oh) * outW + ow;
                    sum += input[in_idx] * grad_out[go_idx];
                }
            }
        }
    }
    grad_w[idx] = sum;
}
