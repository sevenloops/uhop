// uhop/kernels/opencl/depthwise_conv2d.cl
// Depthwise convolution (groups = in_channels)

__kernel void
depthwise_conv2d(__global const float *input, __global const float *weight,
                 __global const float *bias, __global float *output,
                 const int N, const int C, const int H, const int W,
                 const int KH, const int KW, const int stride_h,
                 const int stride_w, const int pad_h, const int pad_w) {

  int ow = get_global_id(0);
  int oh = get_global_id(1);
  int c = get_global_id(2) % C;
  int n = get_global_id(2) / C;

  int outH = (H + 2 * pad_h - KH) / stride_h + 1;
  int outW = (W + 2 * pad_w - KW) / stride_w + 1;

  if (n >= N || c >= C || oh >= outH || ow >= outW)
    return;

  float sum = (bias != 0) ? bias[c] : 0.0f;
  for (int kh = 0; kh < KH; ++kh) {
    for (int kw = 0; kw < KW; ++kw) {
      int ih = oh * stride_h + kh - pad_h;
      int iw = ow * stride_w + kw - pad_w;
      if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        int in_idx = ((n * C + c) * H + ih) * W + iw;
        int w_idx = (c * KH + kh) * KW + kw;
        sum += input[in_idx] * weight[w_idx];
      }
    }
  }
  int out_idx = ((n * C + c) * outH + oh) * outW + ow;
  output[out_idx] = sum;
}
